#ifndef TOK_H
#define TOK_H

#include "struct.h"

BPE *train(uint8_t *data, int len, int max_vocab, Pool *pool);
size_t tok_mem(int len, int max_vocab);
TokenTable *results(BPE *bpe, Pool *pool);

size_t tok_mem(int len, int max_vocab);
BPE *train(uint8_t *data, int len, int max_vocab, Pool *pool);
TokenTable *results(BPE *bpe, Pool *pool);
void tok_ingest(BPE *bpe, Pool *pool);
void tok_train(BPE *bpe, Pool *pool);
void tok_next(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, Pool *pool);
TokenInfo *tok_init(Vocab *vocab, uint16_t left, uint16_t right, Pool *pool);
void tok_activate(Vocab *vocab, TokenInfo *info);
void tok_dec(TokenHeap *heap, Vocab *vocab, TokenInfo *info);
void tok_define_char(Vocab *vocab, uint16_t c, Pool *pool);
TokenInfo *tok_define(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, uint16_t left, uint16_t right, Pool *pool);
Token *tok_push(TokenHashMap *map, TokenHeap *heap, Vocab *vocab, Token *old, uint16_t left, uint16_t right, Pool *pool);
Bucket *th_lookup(TokenHashMap *map, uint32_t key);
void th_set(TokenHashMap *map, uint16_t left, uint16_t right, TokenInfo *val, Pool *pool);
TokenInfo *th_get(TokenHashMap *map, uint16_t left, uint16_t right);
Bucket *th_newbucket(Pool *pool);
TokenHashMap *th_init(int max_bigrams, Pool *pool);
TokenHeap *tq_init(int max_bigrams, Pool *pool);
void swap(TokenHeap *heap, int c, int p);
void tq_siftup(TokenHeap *heap, int start);
void tq_siftdown(TokenHeap *heap, int start);
void tq_push(TokenHeap *heap, TokenInfo *info);
TokenInfo *tq_pop(TokenHeap *heap);
void tq_heapify(TokenHeap *heap);
Vocab *vc_init(int max_bigrams, Pool *pool);
size_t vc_push(Vocab *vocab, TokenInfo *info);
TokenInfo *vc_get(Vocab *vocab, Token *tok);
BPE *bpe_init(uint8_t *data, int len, int max_vocab, Pool *pool);

#endif