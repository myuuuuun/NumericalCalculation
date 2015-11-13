#include <stdio.h>

int main(void){
  float x = 0.1, y = 0.7;
  
  // float型の0.1は真の値よりも少し大きい
  printf("%.20f\n", x);

  // float型の0.7は真の値よりも少し小さい
  printf("%.20f\n", y);
  return 0;
}
