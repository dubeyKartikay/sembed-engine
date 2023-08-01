//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long int MAX_SIZE = 2000;         // max length of strings
const long long int  N = 40;                  // number of closest words that will be shown
const long long int max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  FILE *f_out;
  FILE *f_word;
  char file_name[MAX_SIZE];
  char file_out[MAX_SIZE];
  char file_word[MAX_SIZE];
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./distance <FILE_IN> <FILE_OUT_VEC> <FILE_OUT_WORD>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  strcpy(file_out, argv[2]);
  strcpy(file_word, argv[3]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  f_out = fopen(file_out,"wb");
  f_word = fopen(file_word,"wb");
  if(f_out == NULL){
    printf("Unable to open output file(vector)");
    return -1;
  }
  if(f_word == NULL){
    printf("Unable to open output file(words)");
    return -1;
  }
  fwrite(&words,sizeof(words),1,f_out);
  long long int tot_size = size+1;
  fwrite(&(tot_size),sizeof(tot_size),1,f_out);
  for (float i = 0; i < words; i++)
  {
    // fprintf(f_out,"%lld",i);
    fwrite(&i,sizeof(i),1,f_out);
    fwrite(&M[((int)i)*size],sizeof(float),size,f_out);
    // for ( a = 0; a < size; a++)
    // {
    //   // fprintf(f_out,"%f",M[a+i*size]);
    //   fwrite(M[a+i*size],sizeof(M[a+i*size]),1,f_out);
    // }
    fwrite(&vocab[((int)i)*max_w],max_w,1,f_word);
    
  }
  
  // //Code added by Thomas Mensink
  // //output the vectors of the binary format in text

  // printf("%lld %lld #File: %s\n",words,size,file_name);
  // for (a = 0; a < 1; a++){
  //   printf("%s ",&vocab[a * max_w]);
  //   for (b = 0; b< size; b++){ printf("%f ",M[a*size + b]); }
  //   printf("\b\b\n");
  // }  

  return 0;
}