#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  // N=16個の全粒子のデータをレジスタにロード
  __m512 x_j = _mm512_loadu_ps(x);
  __m512 y_j = _mm512_loadu_ps(y);
  __m512 m_j = _mm512_loadu_ps(m);
  __m512 zero = _mm512_setzero_ps();

  for(int i=0; i<N; i++) {
    // 粒子iの座標を16要素すべてにコピーしてレジスタにセット
    __m512 x_i = _mm512_set1_ps(x[i]);
    __m512 y_i = _mm512_set1_ps(y[i]);

    // rx = x[i] - x[j]
    __m512 rx = _mm512_sub_ps(x_i, x_j);
    // ry = y[i] - y[j]
    __m512 ry = _mm512_sub_ps(y_i, y_j);

    // r^2 = rx*rx + ry*ry
    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512 r2 = _mm512_add_ps(rx2, ry2);

    // i != j の判定をmask operation
    // 自身との距離の2乗は0になるため、0より大きい要素のみを有効とする
    __mmask16 mask = _mm512_cmp_ps_mask(r2, zero, _MM_CMPINT_GT);

    // 1/r
    __m512 inv_r = _mm512_rsqrt14_ps(r2);

    // 1 / (r*r*r) = inv_r * inv_r * inv_r
    __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r));

    // rx * m[j] / (r*r*r)
    __m512 dfx = _mm512_mul_ps(rx, _mm512_mul_ps(m_j, inv_r3));
    // ry * m[j] / (r*r*r)
    __m512 dfy = _mm512_mul_ps(ry, _mm512_mul_ps(m_j, inv_r3));

    // マスクを用いて、i != j の要素のみを抽出
    dfx = _mm512_mask_blend_ps(mask, zero, dfx);
    dfy = _mm512_mask_blend_ps(mask, zero, dfy);

    // レジスタ内の16要素をすべてリダクションしてfx[i], fy[i] から引く
    fx[i] -= _mm512_reduce_add_ps(dfx);
    fy[i] -= _mm512_reduce_add_ps(dfy);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}