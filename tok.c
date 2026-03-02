#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
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

BPE *train(uint8_t *data, int len, int max_vocab, Pool *pool) {
    BPE *bpe = bpe_init(data, len, max_vocab, pool);
    tok_ingest(bpe, pool);
    tok_train(bpe, pool);
    return bpe;
}

TokenTable *results(BPE *bpe, Pool *pool) {
    TokenTable *table = palloc(pool, sizeof(TokenTable));

    int n_pairs = bpe->vocab->used;
    int n_vocab = bpe->vocab->active;
    char **tokens = palloc(pool, sizeof(char *) * n_vocab);
    int *lens = palloc(pool, sizeof(int) * n_vocab);
    TokenInfo **by_id = palloc(pool, sizeof(TokenInfo *) * n_vocab);
    for (int i = 0, j = 0; i < n_pairs && j < n_vocab; i++) {
        TokenInfo *info = bpe->vocab->tokens[i];
        if (!info->active) continue;
        by_id[info->id] = info;
        j++;
    }
    for (int i = 0; i < n_vocab; i++) {
        TokenInfo *info = by_id[i];
        if (!info->active) continue;
        if (i < 256) {
            tokens[i] = palloc(pool, sizeof(char) * 2);
            tokens[i][0] = (char)info->left;
            tokens[i][1] = '\0';
            lens[i] = 1;
        } else {
            char *c1 = tokens[info->left];
            char *c2 = tokens[info->right];
            int l1 = lens[info->left];
            int l2 = lens[info->right];

            tokens[i] = palloc(pool, sizeof(char) * (l1 + l2 + 1));
            lens[i] = l1 + l2;

            memcpy(tokens[i], c1, l1);
            memcpy(tokens[i] + l1, c2, l2);
            tokens[i][l1 + l2] = '\0';
        }
    }
    table->tokens = tokens;
    table->lens = lens;
    table->count = n_vocab;
    return table;
}

void tok_ingest(BPE *bpe, Pool *pool) {
    for (int c = 0; c <= UCHAR_MAX; c++) tok_define_char(bpe->vocab, c, pool);

    Token *prev = NULL;
    Token *cur = NULL;

    for (int i = 0; i < bpe->len - 1; i++) {
        cur = tok_push(bpe->map, bpe->heap, bpe->vocab, NULL, (uint16_t)bpe->data[i], (uint16_t)bpe->data[i + 1], pool);
        if (prev) prev->next = cur;
        cur->prev = prev;
        prev = cur;
    }

    tq_heapify(bpe->heap);
}

void tok_train(BPE *bpe, Pool *pool) {
    while (bpe->vocab->active < bpe->max_vocab) {
        printf("%zu\n", bpe->vocab->active);
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