#include <stdio.h>
#include <math.h>
#define EPSILON 1.0e-6

int main(void){

  float a, b, c;
  a = 0.6;
  b = 0.4;
  c = 0.2;
  printf("%s\n", fabs(a-b-c)<EPSILON ? "a-b == c" : "a-b != c");

  return 0;
}
