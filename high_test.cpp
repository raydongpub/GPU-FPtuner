#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <qd/dd_real.h>

int main()
{
  int n;
  int i;
  int j;
  double *b;
  double *c;
  double *d;
  double err;
  double err_type[3];
  n = 10;
  double *a = new double [n];
  b = (new double [n]);
  c = (new double [n]);
  d = (new double [n]);
  
#pragma 3 3 0 2 0 1 0 0 3 
  for (i = 0; i < n; i++) {
    a[i] = 1.2 + ((double )i);
    b[i] = 2.3 + ((double )i) - 0.5;
    d[i] = 0.0;
  }
  
#pragma tuning
  
#pragma 4 3 0 2 0 1 0 0 3 
  for (i = 0; i < n; i++) {
    c[i] = (a[i]+d[i])*a[i] + b[i] + d[i];
    d[i] = c[i] + a[i];
    b[i] = d[i] + a[i];
  }
  
#pragma 2 3 0 1 0 0 0 2 1 
  for (i = 0; i < n / 2; i++) {
    
#pragma 5 3 0 2 0 0 2 1 3 
    for (j = 0; j < 2; j++) {
      d[i] = c[i] * a[i] * 2 * b[i] - b[i] + a[i];
    }
    d[i] = d[i] / c[i];
  }
  (std::cout<<"The result array: ")<<"\n";
  
#pragma 1 3 0 2 0 1 0 0 0 
  for (i = 0; i < n; i++) 
    printf("%lf ",d[i]);
  std::cout<<"\n";
  delete []a;
  delete []b;
  delete []c;
  delete []d;
}
