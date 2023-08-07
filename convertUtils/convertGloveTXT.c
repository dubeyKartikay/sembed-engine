#include <bits/types/FILE.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc , char ** argv){
  char input_file[100];
  char output_file[100];
  if(argc < 5){
    printf("USAGE ./convertGloveTXT <PATH_TO_INPUT_FILE> <PATH_TO_OUTPUT_VEC_FILE> <PATH_TO_OUTPUT_WORD_FILE> <N> <DIMENTIONS>");
    return -1;
  }
  strcpy(input_file,argv[1]);
  strcpy(output_file,argv[2]);
    
  FILE * p = fopen(input_file, "r");
  FILE * o = fopen(output_file, "wb");
  if(p==NULL){
    printf("Unable to open the input file");
    return -1;
  } 
  if(o==NULL){
    printf("Unable to open the output file");
    return -1;
  }
  char buf[100];
  long long int n = atoll(argv[4]);
  long long int dimentions = atoll(argv[5]);
  printf("%lld\n",n);
  printf("%lld\n",dimentions);
  fwrite(&n, sizeof(n), 1, o);
  long long int tot_dim = dimentions + 1;
  printf("%lld",tot_dim);
  fwrite(&tot_dim, sizeof(tot_dim), 1, o);
  float words = 0;
  

  while (words < n) {
    float vector[dimentions + 1];
    char word[100];
    fscanf(p, "%s",word);
    
    vector[0] = words;
    for (int i = 1; i < dimentions + 1; i++) {
      fscanf(p,"%f",vector+i);
    }
    fwrite(vector, sizeof(float), dimentions + 1, o);
    words++;
  }
  fclose(p);
  fclose(o);
}
