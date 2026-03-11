#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include "struct.h"
#include "ops.h"
#include "tok.h"

size_t tok_mem(int len, int max_vocab) {
   size_t max_bigrams = (size_t)(250.0 * pow(len, 0.6));

    size_t token_size = sizeof(Token) * len;
    size_t bigram_size = sizeof(TokenInfo) * max_bigrams;
    size_t vocab_size = sizeof(Vocab) + sizeof(TokenInfo *) * max_bigrams;
    size_t heap_size = sizeof(TokenHeap) + sizeof(TokenInfo *) * max_bigrams;
    size_t hash_size = sizeof(TokenHashMap) + sizeof(Bucket) * round_up_pow2((size_t)max_bigrams);
    size_t table_size = sizeof(TokenTable) + (sizeof(char *) + sizeof(char) * 4 + sizeof(int)) * max_vocab;
    
    return token_size + bigram_size + vocab_size + heap_size + hash_size + table_size + sizeof(BPE);
}

unsigned char nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

char hex_to_byte(char high, char low) {
    return (char)((nibble(high) << 4) | nibble(low));
}

TokenTable *from_file(char *vocab, char *merges, Pool *pool) {
    int fd = open(merges, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    int n = (int)st.st_size;
    char *data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    int nvocab = 256;
    for (int i = 0; i < n; i++) {
        if (data[i] == '\n') nvocab++;
    }

    TokenTable *table = palloc(pool, sizeof(TokenTable));
    table->count = nvocab;
    table->merges = palloc(pool, sizeof(uint16_t) * 2 * (nvocab - 255));
    table->tokens = palloc(pool, sizeof(char *) * nvocab);
    table->lens = palloc(pool, sizeof(int) * nvocab);
    
    char buf[16];
    for (int i = 0, a = 0; i < n; a++) {
        for (int j = 0;; i++, j++) {
            if (data[i] == ' ') {
                buf[j] = '\0';
                i++;
                break;
            }
            buf[j] = data[i];
        }
        uint16_t l = (uint16_t)strtol(buf, NULL, 10);
        for (int j = 0;; i++, j++) {
            if (i == n || data[i] == '\n') {
                buf[j] = '\0';
                i++;
                break;
            }
            buf[j] = data[i];
        }
        uint16_t r = (uint16_t)strtol(buf, NULL, 10);
        table->merges[a][0] = l;
        table->merges[a][1] = r;
    }
    munmap(data, st.st_size);
    close(fd);

    fd = open(vocab, O_RDONLY);
    fstat(fd, &st);
    n = (int)st.st_size;
    data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    for (int i = 0, a = 0; i < n; a++) {
        int k = 0;
        for (int j = 0; i + j != n && data[i + j] != '\n'; j++) {
            if (data[i + j] == '<') k++;
        }
        table->tokens[a] = palloc(pool, sizeof(char) * (k + 1));
        table->lens[a] = k;
        for (int j = 0; j < k; j++, i += 6) {
            table->tokens[a][j] = hex_to_byte(data[i+3], data[i+4]);
        }
        table->tokens[a][k] = '\0';
        i++;
    }

    munmap(data, st.st_size);
    close(fd);

    return table;
}

Tokenized *tokenize(char *data, int len, TokenTable *table, Pool *pool) {
    Tokenized *res = palloc(pool, sizeof(Tokenized));
    size_t off = pmark(pool);

    TT *head = palloc(pool, sizeof(TT));
    head->id = (uint16_t)((unsigned char)data[0]);
    head->prev = NULL;
    TT *cur = head;
    for (int i = 1; i < len; i++) {
        TT *nxt = palloc(pool, sizeof(TT));
        nxt->id = (uint16_t)((unsigned char)data[i] - 1);
        nxt->next = NULL;
        nxt->prev = cur;
        cur->next = nxt;
        cur = nxt;
    }
    for (uint16_t i = 255; i < table->count; i++) {
        uint16_t l = table->merges[i - 255][0];
        uint16_t r = table->merges[i - 255][1];
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

BPE *bpe_train(uint8_t *data, int len, int max_vocab, Pool *pool) {
    BPE *bpe = bpe_init(data, len, max_vocab, pool);
    tok_ingest(bpe, pool);
    tok_train(bpe, pool);
    return bpe;
}

int cmp(const void *e1, const void *e2) {
    TokenInfo *a = *(TokenInfo **)e1;
    TokenInfo *b = *(TokenInfo **)e2;
    return (int)((a->id > b->id) - (a->id < b->id));
}

TokenTable *results(BPE *bpe, Pool *pool) {
    TokenTable *table = palloc(pool, sizeof(TokenTable));

    table->count = bpe->vocab->active;
    table->tokens = palloc(pool, sizeof(char *) * table->count);
    table->lens = palloc(pool, sizeof(int) * table->count);
    table->merges = palloc(pool, sizeof(uint16_t) * 2 * (table->count - 255));

    TokenInfo *ids[table->count];

    for (int i = 0, j = 0; j < table->count; i++) {
        TokenInfo *info = bpe->vocab->tokens[i];
        if (!info->active) continue;
        ids[j] = info;
        j++;
    }

    qsort(ids, table->count, sizeof(TokenInfo *), cmp);

    for (int i = 0; i < table->count; i++) {
        TokenInfo *info = ids[i];
        if (i < 255) {
            table->tokens[i] = palloc(pool, sizeof(char) * 2);
            table->tokens[i][0] = (char)info->left;
            table->tokens[i][1] = '\0';
            table->lens[i] = 1;
        } else {
            char *c1 = table->tokens[info->left];
            char *c2 = table->tokens[info->right];
            int l1 = table->lens[info->left];
            int l2 = table->lens[info->right];

            printf("%s %s\n", c1, c2);
            table->tokens[i] = palloc(pool, sizeof(char) * (l1 + l2 + 1));
            table->lens[i] = l1 + l2;

            memcpy(table->tokens[i], c1, l1);
            memcpy(table->tokens[i] + l1, c2, l2);
            table->tokens[i][l1 + l2] = '\0';

            table->merges[i - 255][0] = info->left;
            table->merges[i - 255][1] = info->right;
        }

    }
    return table;
}

void tok_ingest(BPE *bpe, Pool *pool) {
    for (int c = 1; c <= UCHAR_MAX; c++) tok_define_char(bpe->vocab, c, pool);

    Token *prev = NULL;
    Token *cur = NULL;

    for (int i = 0; i < bpe->len - 1; i++) {
        cur = tok_push(bpe->map, bpe->heap, bpe->vocab, NULL, (uint16_t)bpe->data[i] - 1, (uint16_t)bpe->data[i + 1] - 1, pool);
        if (prev) prev->next = cur;
        cur->prev = prev;
        prev = cur;
    }

    tq_heapify(bpe->heap);
}

void tok_train(BPE *bpe, Pool *pool) {
    while (bpe->vocab->active < bpe->max_vocab) {
        tok_next(bpe->map, bpe->heap, bpe->vocab, pool);
    }
}

void tok_next(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, Pool *pool) {
    TokenInfo *info = tq_pop(heap), *cur_info;
    Token *tok = info->head;
    Token *prev, *next, *tmp;

    tok_activate(vocab, info);

    while (tok != NULL) {
        prev = tok->prev;
        next = tok->next;
        if (prev) {
            cur_info = vc_get(vocab, prev);

            if (prev->prev_pair) prev->prev_pair->next_pair = prev->next_pair;
            else cur_info->head = prev->next_pair;

            if (prev->next_pair) prev->next_pair->prev_pair = prev->prev_pair;
            else cur_info->tail = prev->prev_pair;

            tmp = prev->prev;

            tok_push(map, heap, vocab, prev, cur_info->left, info->id, pool);

            prev->prev = tmp;
            prev->next = next;
            if (tmp) tmp->next = prev;

            tok_dec(heap, vocab, cur_info);
            tq_siftup(heap, vc_get(vocab, prev)->heap_idx);
        }
        if (next) {
            cur_info = vc_get(vocab, next);

            if (next->prev_pair) next->prev_pair->next_pair = next->next_pair;
            else cur_info->head = next->next_pair;
            if (next->next_pair) next->next_pair->prev_pair = next->prev_pair;
            else cur_info->tail = next->prev_pair;

            tmp = next->next;

            tok_push(map, heap, vocab, next, info->id, cur_info->right, pool);

            next->prev = prev;
            next->next = tmp;
            if (prev) prev->next = next;
            if (tmp) tmp->prev = next;

            tok_dec(heap, vocab, cur_info);
            tq_siftup(heap, vc_get(vocab, next)->heap_idx);
        }
        tok = tok->next_pair;
    }
}

TokenInfo *tok_init(Vocab *vocab, uint16_t left, uint16_t right, Pool *pool) {
    TokenInfo *info = (TokenInfo *)palloc(pool, sizeof(TokenInfo));
    info->head = NULL;
    info->tail = NULL;
    info->id = -1;
    info->idx = vc_push(vocab, info);
    info->left = left;
    info->right = right;
    info->count = 0;
    info->active = 0;
    info->heap_idx = -1;
    return info;
}

void tok_activate(Vocab *vocab, TokenInfo *info) {
    info->active = 1;
    info->id = vocab->active++;
}

void tok_dec(TokenHeap *heap, Vocab *vocab, TokenInfo *info) {
    info->count--;
    if (info->heap_idx >= 0 && info->heap_idx < heap->used) {
        tq_siftdown(heap, info->heap_idx);
    }
}

void tok_define_char(Vocab *vocab, uint16_t c, Pool *pool) {
    TokenInfo *info = tok_init(vocab, c, c, pool);
    tok_activate(vocab, info);
}

TokenInfo *tok_define(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, uint16_t left, uint16_t right, Pool *pool) {
    TokenInfo *info = tok_init(vocab, left, right, pool);
    th_set(map, left, right, info, pool);
    tq_push(heap, info);
    return info;
}

Token *tok_push(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, Token *old, uint16_t left, uint16_t right, Pool *pool) {
    Token *new = old ? old : (Token *)palloc(pool, sizeof(Token));
    TokenInfo *info = th_get(map, left, right);

    if (info == NULL) {
        info = tok_define(map, heap, vocab, left, right, pool);
        info->head = new;
        info->tail = new;
        new->prev_pair = NULL;
    } else {
        Token *tail = info->tail;
        info->tail = new;
        new->prev_pair = tail;
        if (tail) {
            tail->next_pair = new;
        } else {
            info->head = new;
        }
    }

    new->idx = info->idx;
    new->next_pair = NULL;
    new->prev = NULL;
    new->next = NULL;

    info->count++;

    return new;
}

Bucket *th_lookup(TokenHashMap *map, uint32_t key) {
    return &map->buckets[map->hash(key) & (map->size - 1)];
}

void th_set(TokenHashMap *map, uint16_t left, uint16_t right, TokenInfo *val, Pool *pool) {
    uint32_t key = (uint32_t)left << 16 | (uint32_t)right;
    Bucket *cur = th_lookup(map, key);
    while (cur->key != UINT32_MAX) {
        if (cur->key == key) {
            cur->val = val;
            return;
        }
        cur = cur->next;
    }
    cur->key = key;
    cur->val = val;
    cur->next = th_newbucket(pool);
}

TokenInfo *th_get(TokenHashMap *map, uint16_t left, uint16_t right) {
    uint32_t key = (uint32_t)left << 16 | (uint32_t)right;
    Bucket *cur = th_lookup(map, key);
    while (cur->key != key) {
        if (cur->key == UINT32_MAX) return NULL;
        cur = cur->next;
    }
    return cur->val;
}

Bucket *th_newbucket(Pool *pool) {
    Bucket *empty_bucket = (Bucket *)palloc(pool, sizeof(Bucket));
    empty_bucket->key = UINT32_MAX;
    empty_bucket->val = NULL;
    empty_bucket->next = NULL;
    return empty_bucket;
}

TokenHashMap *th_init(int max_bigrams, Pool *pool) {
    TokenHashMap *map = (TokenHashMap *)palloc(pool, sizeof(TokenHashMap));
    map->used = 0;
    map->size = round_up_pow2((size_t)max_bigrams);
    map->buckets = (Bucket *)palloc(pool, sizeof(Bucket) * map->size);
    map->hash = &mueller;
    
    fill(map->buckets, th_newbucket(pool), map->size, sizeof(Bucket));

    return map;
}

TokenHeap *tq_init(int max_bigrams, Pool *pool) {
    TokenHeap *heap = (TokenHeap *)palloc(pool, sizeof(TokenHeap));
    heap->tokens = (TokenInfo **)palloc(pool, sizeof(TokenInfo *) * max_bigrams);
    memset(heap->tokens, 0, sizeof(TokenInfo *) * max_bigrams);
    heap->size = max_bigrams;
    heap->used = 0;
    return heap;
}

void swap(TokenHeap *heap, int c, int p) {
    TokenInfo *tmp = heap->tokens[c];
    heap->tokens[c] = heap->tokens[p];
    heap->tokens[p] = tmp;
    heap->tokens[c]->heap_idx = c;
    heap->tokens[p]->heap_idx = p;
}

void tq_siftup(TokenHeap *heap, int start) {
    for (int i = start; i > 0;) {
        int p = (i - 1) / 2;
        if (heap->tokens[i]->count > heap->tokens[p]->count) {
            swap(heap, i, p);
            i = p;
        } else {
            break;
        }
    }
}

void tq_siftdown(TokenHeap *heap, int start) {
    for (int i = start;;) {
        int c1 = i * 2 + 1;
        int c2 = i * 2 + 2;
        int max = i;

        if (c1 < heap->used && heap->tokens[c1]->count > heap->tokens[max]->count) max = c1;
        if (c2 < heap->used && heap->tokens[c2]->count > heap->tokens[max]->count) max = c2;
        if (max == i) return;

        swap(heap, max, i);
        i = max;
    }
}

void tq_push(TokenHeap *heap, TokenInfo *info) {
    assert(heap->used < heap->size && "heap overflow");
    heap->tokens[heap->used] = info;
    info->heap_idx = heap->used;
    tq_siftup(heap, heap->used++);
}

TokenInfo *tq_pop(TokenHeap *heap) {
    if (heap->used == 0) return NULL;
    TokenInfo *info = heap->tokens[0];
    swap(heap, 0, --heap->used);
    tq_siftdown(heap, 0);
    return info;
}

void tq_heapify(TokenHeap *heap) {
    for (int i = heap->used / 2 - 1; i >= 0; i--) {
        tq_siftdown(heap, i);
    }
}

Vocab *vc_init(int max_bigrams, Pool *pool) {
    Vocab *vocab = (Vocab *)palloc(pool, sizeof(Vocab));
    vocab->tokens = (TokenInfo **)palloc(pool, sizeof(TokenInfo *) * max_bigrams);
    vocab->size = max_bigrams;
    vocab->used = 0;
    vocab->active = 0;
    return vocab;
}

size_t vc_push(Vocab *vocab, TokenInfo *info) {
    assert(vocab->used < vocab->size && "vocab overflow");
    vocab->tokens[vocab->used] = info;
    return vocab->used++;
}

TokenInfo *vc_get(Vocab *vocab, Token *tok) {
    return vocab->tokens[tok->idx];
}

BPE *bpe_init(uint8_t *data, int len, int max_vocab, Pool *pool) {
    BPE *bpe = (BPE *)palloc(pool, sizeof(BPE));
    size_t max_bigrams = (size_t)(250.0 * pow(len, 0.6));

    bpe->data = data;
    bpe->len = len;
    bpe->max_vocab = max_vocab;
    bpe->map = th_init(max_bigrams, pool);
    bpe->heap = tq_init(max_bigrams, pool);
    bpe->vocab = vc_init(max_bigrams, pool);

    return bpe;
}