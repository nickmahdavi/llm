#ifndef IO_H
#define IO_H

#include "struct.h"
#include "tok.h"

TokenTable *train(char *data, int len, int max_vocab, Pool *pool);
TokenTable *train_from_file(char *file, int max_vocab, Pool *pool);

void write_vocab(char *vocab, char *merges, TokenTable *table);
TokenTable *load_vocab(char *vocab, char *merges, Pool *pool);

Tokenized *tokenize_file(char *file, TokenTable *table, Pool *pool);
Tokenized *encode(char *data, int len, TokenTable *table, Pool *pool);
char *decode(Tokenized *tokenized, TokenTable *table, Pool *pool);

#endif