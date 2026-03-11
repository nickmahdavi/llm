#include <limits.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tok.h"
#include "io.h"

unsigned char nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

char hex_to_byte(char high, char low) {
    return (char)((nibble(high) << 4) | nibble(low));
}

TokenTable *train(char *data, int len, int max_vocab, Pool *pool) {
    uint8_t *bpe_data = (uint8_t *)data;
    BPE *bpe = bpe_init(bpe_data, len, max_vocab, pool);
    tok_train(bpe, pool);
    return results(bpe, pool);
}

TokenTable *train_from_file(char *file, int max_vocab, Pool *pool) {
    FILE *fp = fopen(file, "r");
    fseek(fp, 0L, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    int fd = fileno(fp);
    char *data = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    TokenTable *table = train(data, n, max_vocab, pool);
    munmap(data, n);
    fclose(fp);
    return table;
}

Tokenized *tokenize_file(char *file, TokenTable *table, Pool *pool) {
    FILE *fp = fopen(file, "r");
    fseek(fp, 0L, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    int fd = fileno(fp);
    char *data = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    Tokenized *out = encode(data, (int)n, table, pool);
    munmap(data, n);
    fclose(fp);
    return out;
}

TokenTable *load_vocab(char *vocab, char *merges, Pool *pool) {
    FILE *vocab_fp = fopen(vocab, "r");
    fseek(vocab_fp, 0L, SEEK_END);
    long vocab_n = ftell(vocab_fp);
    fseek(vocab_fp, 0L, SEEK_SET);
    int vocab_fd = fileno(vocab_fp);
    char *vocab_data = mmap(NULL, vocab_n, PROT_READ, MAP_PRIVATE, vocab_fd, 0);

    int count = 0;
    for (int i = 0; i < vocab_n; i++) {
        if (vocab_data[i] == '\n') count++;
    }
    count++;

    TokenTable *table = palloc(pool, sizeof(TokenTable));
    table->count = count;
    table->merges = palloc(pool, sizeof(uint16_t) * 2 * (count - 256));
    table->tokens = palloc(pool, sizeof(char *) * count);
    table->lens = palloc(pool, sizeof(int) * count);

    for (long i = 0, a = 0; i < vocab_n; a++) {
        int k = 0;
        for (long j = 0; i + j != vocab_n && vocab_data[i + j] != '\n'; j++) {
            if (vocab_data[i + j] == '<') k++;
        }
        table->tokens[a] = palloc(pool, sizeof(char) * (k + 1));
        table->lens[a] = k;
        for (int j = 0; j < k; j++, i += 6) {
            table->tokens[a][j] = hex_to_byte(vocab_data[i+3], vocab_data[i+4]);
        }
        table->tokens[a][k] = '\0';
        i++;
    }

    munmap(vocab_data, vocab_n);
    fclose(vocab_fp);

    FILE *merges_fp = fopen(merges, "r");
    fseek(merges_fp, 0L, SEEK_END);
    long merges_n = ftell(merges_fp);
    fseek(merges_fp, 0L, SEEK_SET);
    long merges_fd = fileno(merges_fp);
    char *merges_data = mmap(NULL, merges_n, PROT_READ, MAP_PRIVATE, merges_fd, 0);

    char buf[16];
    for (long i = 0, a = 0; i < merges_n; a++) {
        for (long j = 0;; i++, j++) {
            if (merges_data[i] == ' ') {
                buf[j] = '\0';
                i++;
                break;
            }
            buf[j] = merges_data[i];
        }
        uint16_t l = (uint16_t)strtol(buf, NULL, 10);
        for (long j = 0;; i++, j++) {
            if (i == merges_n || merges_data[i] == '\n') {
                buf[j] = '\0';
                i++;
                break;
            }
            buf[j] = merges_data[i];
        }
        uint16_t r = (uint16_t)strtol(buf, NULL, 10);
        table->merges[a][0] = l;
        table->merges[a][1] = r;
    }

    munmap(merges_data, merges_n);
    fclose(merges_fp);

    return table;
}

void write_vocab(char *vocab, char *merges, TokenTable *table) {
    FILE *vocab_fp = fopen(vocab, "w");
    FILE *merges_fp = fopen(merges, "w");

    for (int i = 0; i < table->count; i++) {
        if (i) fprintf(vocab_fp, "\n");
        for (int j = 0; j < table->lens[i]; j++) {
            fprintf(vocab_fp, "<0x%02X>", (unsigned char)table->tokens[i][j]);
        }
    }

    for (int i = 0; i < table->count - 256; i++){ 
        if (i) fprintf(merges_fp, "\n");
        fprintf(merges_fp, "%hu", table->merges[i][0]);
        fprintf(merges_fp, " ");
        fprintf(merges_fp, "%hu", table->merges[i][1]);
    }

    fclose(vocab_fp);
    fclose(merges_fp);
}

Tokenized *encode(char *data, int len, TokenTable *table, Pool *pool) {
    Tokenized *res = palloc(pool, sizeof(Tokenized));
    size_t off = pmark(pool);

    TT *head = palloc(pool, sizeof(TT));
    head->id = (uint16_t)((unsigned char)data[0]);
    head->prev = NULL;
    TT *cur = head;
    for (int i = 1; i < len; i++) {
        TT *nxt = palloc(pool, sizeof(TT));
        nxt->id = (uint16_t)((unsigned char)data[i]);
        nxt->next = NULL;
        nxt->prev = cur;
        cur->next = nxt;
        cur = nxt;
    }
    for (uint16_t i = 256; i < table->count; i++) {
        uint16_t l = table->merges[i - 256][0];
        uint16_t r = table->merges[i - 256][1];
        for (TT *cur = head->next; cur->next != NULL; cur = cur->next) {
            if (cur->prev->id == l && cur->id == r) {
                cur->prev->id = i;
                cur->prev->next = cur->next;
                cur->next->prev = cur->prev;
            }
        }
    }
    int acc = 0;
    for (TT *cur = head; cur != NULL; cur = cur->next) acc++;
    res->len = acc;

    uint16_t toks[acc];

    int i = 0;
    for (TT *cur = head; cur != NULL; cur = cur->next) toks[i++] = cur->id;

    prollback(pool, off);
    res->tokens = palloc(pool, sizeof(uint16_t) * acc);
    memcpy(res->tokens, toks, sizeof(uint16_t) * acc);
    return res;
}

char *decode(Tokenized *tokenized, TokenTable *table, Pool *pool) {
    int acc = 0;
    for (int i = 0; i < tokenized->len; i++) {
        acc += table->lens[tokenized->tokens[i]];
    }
    char *out = palloc(pool, sizeof(char) * (acc + 1));
    char *ptr = out;
    for (int i = 0; i < tokenized->len; i++) {
        ptr = stpcpy(ptr, table->tokens[tokenized->tokens[i]]);
    }
    return out;
}