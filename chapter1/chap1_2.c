#include <stdio.h>

int main(void){

  float a, b, c;
  a = 0.6;
  b = 0.4;
  c = 0.2;
  printf("%s\n", a-b == c ? "a-b == c" : "a-b != c");
  printf("a-b: %.20f\nc: %.20f\n", a-b, c);

  return 0;
}
