using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;
using MemoryMarshal = System.Runtime.InteropServices.MemoryMarshal;
using BigInteger = System.Numerics.BigInteger;
using StringBuilder = System.Text.StringBuilder;
using LIB340.TEMPLATES;
using LIB340.DataStructure;
/// <summary>
/// C# 競技プログラミング用ライブラリ
/// [使い方]
/// エディタの検索機能でクラスを探す。=> 折り畳んでからクラスをコピペする
///
/// [コーディング規則]
/// クラス•メソッドの説明はsummaryで囲う。
/// 
///
/// 
/// [目次]  ()はスニペット済 
///
/// 数学関連 : MyMath
/// デバッグ : Debug
/// 順列に変換 : IndexConverter
/// Dictionary代わり : Counter
/// (有無向グラフ)
/// (重み有・無UnionFind)
/// (優先キュー) : PriorityQueue
/// ダイクストラ : Dijkstraa
/// 最小全域木 : Kruskal
/// (Modular)
/// 行列
/// 半分全列挙：halffullenumeration
/// トポソ
/// Zアルゴリズム
/// BIT
/// セグ木
/// 遅延評価セグ木
/// 両端キュー :Deque
/// 累積和 : CumulativeSum
/// 座標圧縮 : CoordinateCompression
///
/// その他 DOC
/// DPはパスカルの三角形を意識する
/// </summary>

namespace LIB340
{

    // アルゴリズム データ構造 の順で並べる
    // よく使うものほど前に
    namespace MathLIB
    {
        public static class MyMath
        {
            /// <summary>最大公約数  </summary>
            public static long GCD(long a, long b) => a == 0 ? b : GCD(b % a, a);
            /// <summary>最小公倍数 </summary>
            public static long LCM(long a, long b) => a / GCD(a, b) * b;
            /// <summary>拡張GCD  ax + by = gcd(a,b) </summary>
            public static (long x, long y) extGCD(long a, long b)
            {
                if (a < b)
                {
                    var (x, y) = extGCD(b, a);
                    return (y, x);
                }
                if (b == 0) return (1, 0);
                else
                {
                    var t = extGCD(b, a % b);
                    return (t.y, t.x - a / b * t.y);
                }
            }

            public static bool IsPrime(int num)
            {
                for (int i = 2; i * i <= num; i++)
                {
                    if (num % i == 0) return false;
                }
                return true;
            }

            /// <summary>冪剰余 (a^n % mod).  a <= 10^9 </summary>
            public static long ModPow(long a, long n, long mod)
            {
                long res = 1;
                while (n > 0)
                {
                    if ((n & 1) >= 1) res = res * a % mod;
                    a = a * a % mod;
                    n >>= 1;
                }
                return res;
            }

            /// <summary>
            /// 全順列列挙。
            /// [1,2,3] -> [1,2,3],[2,1,3],[3,1,2],[1,3,2],[2,3,1],[3,2,1]
            /// </summary>
            public static IEnumerable<IEnumerable<T>> Permutation<T>(IEnumerable<T> source)
            {
                var items = source.ToArray();
                yield return items;
                var counter = new int[items.Length];
                var idx = 0;
                var count = 0;
                var fact = 1L;
                for (var i = 1; i <= items.Length; i++) fact *= i;
                while (idx < items.Length)
                {
                    if (counter[idx] < idx)
                    {
                        if (idx % 2 == 0) (items[0], items[idx]) = (items[idx], items[0]);
                        else (items[counter[idx]], items[idx]) = (items[idx], items[counter[idx]]);
                        yield return items;
                        counter[idx]++;
                        count++;
                        idx = 0;
                    }
                    else
                    {
                        counter[idx] = 0;
                        idx++;
                    }
                    if (count == fact) yield break;
                }
            }

            /// <summary>進数変換 </summary>
            public static int[] ConvertBase(long sourse, int b)
            {
                long num = 1;
                while (num <= sourse)
                {
                    num *= b;
                }
                num /= b;
                List<int> ans = new List<int>();
                while (num >= 1)
                {
                    int c = (int)(sourse / num);
                    ans.Add(c);
                    sourse -= num * c;
                    num /= b;
                }
                return ans.ToArray();
            }

            

        }

        //Modしない計算 BigInteger型で返す
        // TODO 
        /// <summary>
        /// BigInterger型の演算を追加します。
        /// </summary>
        class BigIntegerExtension
        {

            const int arysize = 100;
            static BigInteger[] facs = new BigInteger[arysize];
            static int facscount = -1;
            public static BigInteger Fac(BigInteger n)
            {
                facs[0] = 1;
                while (facscount <= n)
                {
                    facs[++facscount + 1] = facs[facscount] * (facscount + 1);
                }
                return facs[(int)n];
            }
            public static BigInteger Fac(BigInteger n, BigInteger r)
            {
                BigInteger ans = n;
                while (n++ < r)
                {
                    ans *= n;
                }
                return ans;
                
            }
            public static BigInteger nCr(BigInteger n, BigInteger r)
            {
                return (n < r) ? 0
                     : (n == r) ? 1
                                : (BigInteger.Max(n, r) <= arysize) ? Fac(n) / (Fac(r) * Fac(n - r))
                                    : Fac(n - r + 1, n) / Fac(r);
            }
            public static BigInteger nPr(BigInteger n, BigInteger r)
            {
                return Fac(n) / Fac(n - r);
            }
        }

        // 1000000007でModした計算
        /// <summary>
        /// 静的な値で剰余した値を返します。
        /// </summary>
        struct Modular
        {
            const int M = 1000000007;
            const int arysize = 2000001;
            long value;
            public Modular(long value = 0) { this.value = value; }
            public override string ToString() { return value.ToString(); }
            public static implicit operator Modular(long a)
            {
                var m = a % M;
                return new Modular((m < 0) ? m + M : m);
            }
            public static Modular operator +(Modular a, Modular b) => a.value + b.value;
            public static Modular operator -(Modular a, Modular b) => a.value - b.value;
            public static Modular operator *(Modular a, Modular b) => a.value * b.value;
            public static Modular operator /(Modular a, Modular b) => a * Pow(b, M - 2);
            public static Modular operator ++(Modular a) => a.value + 1;
            public static Modular operator --(Modular a) => a.value - 1;
            public static Modular Pow(Modular a, long n)
            {
                Modular ans = 1;
                for (; n > 0; n >>= 1, a *= a)
                {
                    if ((n & 1) == 1) ans *= a;
                }
                return ans;
            }
            static int[] facs = new int[arysize];
            static int facscount = -1;
            public static Modular Fac(int n)   //階乗
            {
                facs[0] = 1;
                while (facscount <= n)
                {
                    facs[++facscount + 1] = (int)(Math.BigMul(facs[facscount], facscount + 1) % M);
                }
                return facs[n];
            }
            public static Modular Fac(int r, int n)//記録しない階乗
            {
                int temp = 1;
                for (int i = r; i <= n; i++)
                {
                    temp = (int)(Math.BigMul(temp, i) % M);
                }
                return temp;
            }
            public static Modular Ncr(int n, int r) //nCr
            {
                return (n < r) ? 0
                     : (n == r) ? 1
                                : (Math.Max(n, r) <= arysize) ? Fac(n) / (Fac(r) * Fac(n - r))
                                    : Fac(n - r + 1, n) / Fac(r);
            }
            public static Modular Npr(int n, int r)
            {
                if (Math.Max(n, r) <= arysize) return Fac(n) / Fac(n - r);
                return Fac(n - r + 1, n);
            }
            public static explicit operator int(Modular a)
            {
                return (int)a.value;
            }
        }

        /// <summary>
        /// 値が静的な値で剰余される行列を扱います。
        /// TODO 機能追加
        /// </summary>
        class ModMat
        {
            long[,] Data;
            static readonly long Mod = 1000000007;
            public ModMat(int _size)
            {
                Size = _size;
                Data = new long[Size, Size];
            }
            public ModMat(int[,] _mat)
            {
                Size = _mat.GetLength(0);
                Data = new long[Size, Size];
                Array.Copy(_mat, Data, Size * Size);
            }
            public int Size { get; }
            public long this[int i, int j]
            {
                get => Data[i, j];
                set { Data[i, j] = value; Data[i, j] %= Mod; }
            }
            public static ModMat operator +(ModMat A, ModMat B)
            {
                if (A.Size != B.Size) throw new Exception($"ex at'+' a.size={A.Size} b.size={B.Size}");
                for (int i = 0; i < A.Size; i++)
                {
                    for (int j = 0; j < A.Size; j++)
                    {
                        A[i, j] = (A[i, j] + B[i, j]) % Mod;
                    }
                }
                return A;
            }
            public static ModMat operator -(ModMat A, ModMat B)
            {
                if (A.Size != B.Size) throw new Exception($"ex at'-' a.size={A.Size} b.size={B.Size}");
                for (int i = 0; i < A.Size; i++)
                {
                    for (int j = 0; j < A.Size; j++)
                    {
                        A[i, j] = (A[i, j] - B[i, j] + Mod) % Mod;
                    }
                }
                return A;
            }
            public static ModMat operator *(ModMat A, ModMat B)
            {
                if (A.Size != B.Size) throw new Exception($"ex at'*' a.size={A.Size} b.size={B.Size}");
                int N = A.Size;
                var c = new ModMat(N);
                for (int i = 0; i < N; i++)
                {
                    for (int k = 0; k < N; k++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            c[i, j] = (c[i, j] + A[i, k] * B[k, j]) % Mod;
                        }
                    }
                }
                return c;
            }
            public static ModMat operator *(ModMat A, long b)
            {
                var C = new ModMat(A.Size);
                for (int i = 0; i < A.Size; i++)
                {
                    for (int j = 0; j < A.Size; j++)
                    {
                        C[i, j] = A[i, j] * b % Mod;
                    }
                }
                return C;
            }
            public static ModMat Pow(ModMat A, long n)
            {
                ModMat B = new ModMat(A.Size);
                for (int i = 0; i < A.Size; i++)
                {
                    B[i, i] = 1;
                }
                while (n > 0)
                {
                    if ((n & 1) == 1) B *= A;
                    A = A * A;
                    n >>= 1;
                }
                return B;
            }
        }


    }

    /// <summary>
    /// データ構造
    /// よく使用するもの スニペット済みのもの あまり使わないもの　の順に並べる
    /// </summary>
    namespace DataStructure
    {
        // (スニペット済み)
        /// <summary>
        /// 両端キュー 配列の前後で追加•Pop操作ができます 。
        /// PushFront & PushBack : O(1)
        /// PopFront & PopBack : O(1)
        /// PeekFront & PeekBack : O(1)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class Deque<T> : IEnumerable<T>
        {
            T[] buf;
            int offset, count, cap;
            public int Count { get { return count; } }
            public Deque(IEnumerable<T> collection) : this()
            {
                foreach (var item in collection) PushBack(item);
            }
            public Deque(int cap) { buf = new T[this.cap = cap]; }
            public Deque() { buf = new T[cap = 16]; }
            public T this[int index]
            {
                get { return buf[GetIndex(index)]; }
                set { buf[GetIndex(index)] = value; }
            }
            private int GetIndex(int index)
            {
                if (index >= cap) throw new IndexOutOfRangeException();
                var ret = index + offset;
                return ret >= cap ? ret - cap : ret;
            }
            public T PeekFront() => buf[offset];
            public T PeekBack() => buf[GetIndex(Count - 1)];
            public void PushFront(T item)
            {
                if (count == cap) Extend();
                if (--offset < 0) offset += buf.Length;
                buf[offset] = item;
                ++count;
            }
            public T PopFront()
            {
                if (count == 0) throw new InvalidOperationException("collection is empty");
                --count;
                var ret = buf[offset++];
                if (offset >= cap) offset -= cap;
                return ret;
            }
            public void PushBack(T item)
            {
                if (count == cap) Extend();
                var id = count++ + offset;
                if (id >= cap) id -= cap;
                buf[id] = item;
            }
            public T PopBack()
            {
                if (count == 0) throw new InvalidOperationException("collection is empty");
                return buf[GetIndex(--count)];
            }
            public void Insert(int index, T item)
            {
                if (index > count) throw new IndexOutOfRangeException();
                this.PushFront(item);
                for (int i = 0; i < index; i++)
                    this[i] = this[i + 1];
                this[index] = item;
            }
            public T RemoveAt(int index)
            {
                if (index < 0 || index >= count) throw new IndexOutOfRangeException();
                var ret = this[index];
                for (int i = index; i > 0; i--)
                    this[i] = this[i - 1];
                this.PopFront();
                return ret;
            }
            void Extend()
            {
                T[] newBuffer = new T[cap << 1];
                if (offset > cap - count)
                {
                    var len = buf.Length - offset;
                    Array.Copy(buf, offset, newBuffer, 0, len);
                    Array.Copy(buf, 0, newBuffer, len, count - len);
                }
                else Array.Copy(buf, offset, newBuffer, 0, count);
                buf = newBuffer;
                offset = 0;
                cap <<= 1;
            }

            public IEnumerator<T> GetEnumerator() => Items.ToList().GetEnumerator();

            IEnumerator IEnumerable.GetEnumerator()
            {
                throw new NotImplementedException();
            }

            public T[] Items//デバッグ時に中身を調べるためのプロパティ
            {
                get
                {
                    var a = new T[count];
                    for (int i = 0; i < count; i++)
                        a[i] = this[i];
                    return a;
                }
            }
        }



        /// <summary>
        /// 普通のUnionFind 。
        /// Unite：O(α(n)) αはアッカーマン関数の逆関数 定数の3程度
        /// </summary>
        public class UnionFind
        {
            int n;
            int[] p;
            public UnionFind(int n)
            {
                this.n = n;
                p = Enumerable.Repeat(-1, n).ToArray();
            }
            /// <summary>グループを結合します。 O(α(n)) </summary>
            public bool Unite(int x, int y)
            {
                (x, y) = (Root(x), Root(y));
                if (x != y)
                {
                    if (p[y] < p[x]) (y, x) = (x, y);
                    p[x] += p[y];
                    p[y] = x;
                }
                return x != y;
            }
            /// <summary>同じグループか判別します。 </summary>
            public bool IsSame(int x, int y) => Root(x) == Root(y);
            public int Root(int x) => p[x] < 0 ? x : p[x] = Root(p[x]); //親のノードを探す
            public int GetMem(int x) => -p[Root(x)];
            /// <summary>グループごとに分けたListを出力 </summary>
            public List<int>[] Groups
            {
                get
                {
                    var res = new List<int>[n].Select(_ => new List<int>()).ToArray();
                    for (int i = 0; i < n; i++)
                    {
                        if (p[i] >= 0) res[Root(i)].Add(i);
                        else res[i].Add(i);
                    }
                    return res.Where(vs => vs.Count > 0).ToArray();
                }
            }
        }


        /// <summary>
        /// 重み付きUnionFind。  親ノードに重みのMerge結果をまとめます。
        /// 初期化時に重みを入力する
        /// </summary>
        /// <typeparam name="T">重みの型</typeparam>
        public class WeightedUnionFind<T>
        {
            int n;
            int[] P;
            T[] W;
            Func<T, T, T> Merge;
            public WeightedUnionFind(int size) : this(size, (a, b) => default, new T[size]) { }
            public WeightedUnionFind(int n, Func<T, T, T> merge, T[] init)
            {
                this.n = n;
                P = Enumerable.Repeat(-1, n).ToArray();
                W = init;
                Merge = merge;
            }
            /// <summary>グループを結合します。要素数の大きい方が親になります。 </summary>
            public bool Unite(int x, int y)
            {
                x = Root(x);
                y = Root(y);
                if (x != y)
                {
                    if (P[y] < P[x]) (y, x) = (x, y);
                    P[x] += P[y];
                    P[y] = x;
                    W[x] = Merge(W[x], W[y]);
                }
                return x != y;
            }
            public bool IsSameGroup(int x, int y) => Root(x) == Root(y);
            public int Root(int x) => P[x] < 0 ? x : P[x] = Root(P[x]);
            public int GetMem(int x) => -P[Root(x)];
            public T Weight(int x) => W[x];
            /// <summary>要素の親の重みを取得します。 Set非推奨! </summary>
            public T this[int x] { get => W[Root(x)]; set => W[Root(x)] = value; }
            public List<int>[] Groups
            {
                get
                {
                    var res = new List<int>[n].Select(_ => new List<int>()).ToArray();
                    for (int i = 0; i < n; i++)
                    {
                        if (P[i] >= 0) res[Root(i)].Add(i);
                        else res[i].Add(i);
                    }
                    return res.Where(vs => vs.Count > 0).ToArray();
                }
            }
        }


        /// <summary>
        /// 重み付きUnionFind 子に値を持たせる。(重みはlong型)
        /// マージ時に重みの差異を入力する
        /// </summary>
        public class WeightedUnionFind
        {
            int[] p, rank;
            long[] dW;
            public WeightedUnionFind(int N)
            {
                p = Enumerable.Repeat(-1, N).ToArray();
                rank = new int[N];
                dW = new long[N];
            }
            /// <summary>グループの結合</summary><param name="diff">重みの差異。</param><returns></returns>
            public bool Merge(int x, int y, long diff)
            {
                diff += Weight(x) - Weight(y);
                x = Root(x);
                y = Root(y);
                if (x != y)
                {
                    if (rank[x] < rank[y])
                    {
                        (x, y) = (y, x);
                        diff = -diff;
                    }
                    p[x] += p[y];
                    p[y] = x;
                    dW[y] += diff;
                    if (rank[x] == rank[y]) rank[x]++;
                }
                return x != y;
            }
            int Root(int x) //親から順に重さを更新
            {
                if (p[x] < 0) return x;
                int root = Root(p[x]);
                dW[x] += dW[p[x]];
                return p[x] = root;
            }
            public long Weight(int x)
            {
                Root(x);
                return dW[x];
            }
            public bool IsSame(int x, int y) => Root(x) == Root(y);
            public long GetDiff(int x, int y) => Weight(y) - Weight(x);
            public int GetMem(int x) => -p[Root(x)];
        }



        /// <summary>
        /// セグメントツリー 単更新・範囲検索 ジェネリクス
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class SegTree<T>
        {
            int N;
            T init;
            T[] dat;
            Func<T, T, T> updFunc, qFunc;
            /// <param name="n">要素数</param>
            /// <param name="_init">単位元</param>
            /// <param name="updfunc">子をマージするときの式</param>
            /// <param name="qfunc"><子同士のクエリの結果をマージする式/param>
            public SegTree(int n, T _init, Func<T, T, T> updfunc, Func<T, T, T> qfunc)
            {
                N = 1;
                while (N <= n) N <<= 1;
                dat = new T[2 * N - 1];
                dat.AsSpan().Fill(init);
                updFunc = updfunc;
                qFunc = qfunc;
                init = _init;
            }
            /// <summary>点aにxを追加します。 </summary>
            public void Update(int a, T x)
            {
                a += N - 1;
                dat[a] = x;
                while (a > 0)
                {
                    a = (a - 1) >> 1;
                    dat[a] = updFunc(dat[(a << 1) + 1], dat[(a << 1) + 2]);
                }
            }
            /// <summary>点aのクエリ結果を返します。O(1)</summary>
            public T Query(int a) => dat[N - 1 + a];
            /// <summary>区間[a,b)のクエリ結果を返します。O(logN)</summary>
            public T Query(int a, int b) => Query(a, b, 0, 0, N);
            private T Query(int a, int b, int k = 0, int l = 0, int r = 0)
            {
                if (r <= a || b <= l) return init;
                else if (a <= l && r <= b) return dat[k];
                else
                {
                    T vl = Query(a, b, (k << 1) + 1, l, (l + r) >> 1);
                    T vr = Query(a, b, (k << 1) + 2, (l + r) >> 1, r);
                    return qFunc(vl, vr);
                }
            }
            ///<summary> set => Update(a,value), get => Query(a)</summary>
            public T this[int a] { get => Query(a); set => Update(a, value); }

            public T this[int a, int b] => Query(a, b);
        }



        /// <summary>
        /// 遅延評価セグメントツリー  範囲更新・範囲検索
        /// [a,b)に対する演算
        /// </summary>
        class LazySegTree
        {
            int n;
            long[] Data, Lazy;
            Func<long, long, long> qfunc, updfunc, efunc2, efunc3;
            Func<long, long, long, long> efunc1;
            public LazySegTree(int size, Func<long, long, long> _qfunc, Func<long, long, long> _updfunc, Func<long, long, long, long> _efunc1, Func<long, long, long> _efunc2, Func<long, long, long> _efunc3) : this(new int[size], _qfunc, _updfunc, _efunc1, _efunc2, _efunc3) { }
            public LazySegTree(int[] v) : this(v, (x, y) => x + y, (x, y) => x + y, (x, y, z) => x + y * z, (x, y) => x + y / 2, (x, y) => x + y) { }
            public LazySegTree(int size) : this(new int[size], (x, y) => x + y, (x, y) => x + y, (x, y, z) => x + y * z, (x, y) => x + y / 2, (x, y) => x + y) { }
            public LazySegTree(int[] v, Func<long, long, long> _qfunc, Func<long, long, long> _updfunc, Func<long, long, long, long> _efunc1, Func<long, long, long> _efunc2, Func<long, long, long> _efunc3)
            {
                int size = v.Length;
                qfunc = _qfunc;
                updfunc = _updfunc;
                efunc1 = _efunc1;
                efunc2 = _efunc2;
                efunc3 = _efunc3;
                n = 1; while (n < size) n <<= 1;
                Data = new long[n * 2 - 1];
                Lazy = new long[n * 2 - 1];
                for (int i = 0; i < size; i++)
                {
                    Data[i + n - 1] = v[i];
                }
                for (int i = n - 2; i >= 0; i--)//二段目n-1から(一段目は2n-1)
                {
                    Data[i] = updfunc(Data[i * 2 + 1], Data[i * 2 + 2]);
                }
            }
            void eval(int k, int l, int r)//nodeが呼び出された時に伝達する。
            {
                if (Lazy[k] != default)
                {
                    Data[k] = efunc3(Data[k], Lazy[k]); // Data[k]+Lazy[k]
                    if (r - l > 1)
                    {
                        Lazy[k * 2 + 1] = efunc2(Lazy[k * 2 + 1], Lazy[k]); // Lazy[k]/2
                        Lazy[k * 2 + 2] = efunc2(Lazy[k * 2 + 2], Lazy[k]); // Lazy[k]/2
                    }
                }
                Lazy[k] = default;
            }
            /// <summary>点a に値xを追加します。</summary>
            public void Update(int a, long x) => Update(a, a + 1, x, 0, 0, n);
            /// <summary>区間[a,b) に値xを追加します。</summary>
            public void Update(int a, int b, long x) => Update(a, b, x, 0, 0, n); //[a,b)
            private void Update(int a, int b, long x, int k, int l, int r)
            {
                eval(k, l, r);
                if (b <= l || r <= a) return;
                if (a <= l && r <= b)//完全にl,rが含まれる
                {
                    Lazy[k] = efunc1(Lazy[k], r - l, x); // Lazy[k]+(r-l)*x
                    eval(k, l, r);
                }
                else//どっちか片方範囲外
                {
                    Update(a, b, x, k * 2 + 1, l, (l + r) >> 1);
                    Update(a, b, x, k * 2 + 2, (l + r) >> 1, r);
                    Data[k] = updfunc(Data[k * 2 + 1], Data[k * 2 + 2]);
                }
            }
            /// <summary>点a の値を取得します。</summary>
            public long Query(int a) => Query(a, a + 1, 0, 0, n);
            /// <summary>区間[a,b) の値を取得します。</summary>
            public long Query(int a, int b) => Query(a, b, 0, 0, n);
            private long Query(int a, int b, int k, int l, int r)
            {
                if (b <= l || r <= a) return 0;
                eval(k, l, r);
                if (a <= l && r <= b) return Data[k];
                else
                {
                    var vl = Query(a, b, k * 2 + 1, l, (l + r) >> 1);
                    var vr = Query(a, b, k * 2 + 2, (l + r) >> 1, r);
                    return qfunc(vl, vr);
                }
            }
            public long this[int a] { get => Query(a); set => Update(a, value); }
            public long this[int a, int b] { get => Query(a, b); set => Update(a, b, value); }
        }



        // (スニペット済み)
        /// <summary>
        /// 優先キュー 優先度の高い要素から順に取り出します。
        /// Enqueue : O(logN)
        /// Dequeue : O(logN)
        /// Peek : O(1)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class PriorityQueue<T> : IEnumerable<T>, ICollection, IEnumerable, ICloneable
        {
            List<T> m_heap;
            Comparison<T> Comp;
            public int Count => m_heap.Count;
            public bool IsEmpty => Count == 0;
            public PriorityQueue(IEnumerable<T> source) : this(null, 16, source) { }
            public PriorityQueue(int capacity = 16, IEnumerable<T> source = null) : this(null, capacity, source) { }
            public PriorityQueue(Comparison<T> comp, IEnumerable<T> source) : this(comp, 16, source) { }
            public PriorityQueue(Comparison<T> comp, int capacity = 16, IEnumerable<T> source = null) { this.Comp = comp == null ? (x, y) => Comparer<T>.Default.Compare(x, y) : comp; m_heap = new List<T>(capacity); if (source != null) foreach (var x in source) Enqueue(x); }
            /// <summary>要素を追加します。</summary>
            public void Enqueue(T x)
            {
                var pos = Count;
                m_heap.Add(x);
                while (pos > 0)
                {
                    var p = (pos - 1) / 2;
                    if (Comp(m_heap[p], x) <= 0) break;
                    m_heap[pos] = m_heap[p];
                    pos = p;
                }
                m_heap[pos] = x;
                var que = new Queue<int>();

            }
            /// <summary>先頭の要素を取り出します。(値はキューから削除。)</summary>
            public T Dequeue()
            {
                var value = m_heap[0];
                var x = m_heap[Count - 1];
                m_heap.RemoveAt(Count - 1);
                if (Count == 0) return value;
                var pos = 0;
                while (pos * 2 + 1 < Count)
                {
                    var a = 2 * pos + 1;
                    var b = 2 * pos + 2;
                    if (b < Count && Comp(m_heap[b], m_heap[a]) < 0) a = b;
                    if (Comp(m_heap[a], x) >= 0) break;
                    m_heap[pos] = m_heap[a];
                    pos = a;
                }
                m_heap[pos] = x;
                return value;
            }
            /// <summary>先頭の要素を取得します。 (値はキューに保持。)</summary>
            public T Peek() => m_heap[0];
            public IEnumerator<T> GetEnumerator() { var x = (PriorityQueue<T>)Clone(); while (x.Count > 0) yield return x.Dequeue(); }
            void CopyTo(Array array, int index) { foreach (var x in this) array.SetValue(x, index++); }
            public object Clone() { var x = new PriorityQueue<T>(Comp, Count); x.m_heap.AddRange(m_heap); return x; }
            public void Clear() => m_heap = new List<T>();
            public void TrimExcess() => m_heap.TrimExcess();
            public bool Contains(T item) { return m_heap.Contains(item); } //O(N)
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
            void ICollection.CopyTo(Array array, int index) => CopyTo(array, index);
            bool ICollection.IsSynchronized => false;
            object ICollection.SyncRoot => this;
        }

        
    }


    /// <summary>
    /// グラフを使ったライブラリ群。
    /// </summary>
    namespace GraphExtension
    {
        /// <summary>
        /// グラフ
        /// 追加する辺次第で重み有り無し　変更できます。
        /// 
        /// 隣接リスト(グラフ) サイズの自動拡張機能つき
        /// new時にサイズ指定するとサイズは固定
        /// 辺重複可
        /// </summary>
        /// <typeparam name="TEdge"></typeparam>
        class Graph<TEdge>
        {
            int maxsize = 0;
            private Node<TEdge>[] G;
            public Graph(int size = 1024)
            {
                maxsize = size;
                G = new Node<TEdge>[size].Select(_ => _ = new Node<TEdge>()).ToArray();
            }
            /// <summary>一方向に辺を追加します。</summary>
            /// <param name="edge"></param>
            public void Add(Edge<TEdge> edge)
            {
                while (Math.Max(edge.From, edge.To) >= maxsize) Expand();
                G[edge.From].edges.Add(edge);
            }
            public void Add(int from, int to, TEdge value = default) => Add(new Edge<TEdge> { From = from, To = to, Value = value });
            /// <summary>双方向に辺を追加します。</summary>
            public void AddBoth(Edge<TEdge> edge)
            {
                Add(edge);
                Add(new Edge<TEdge> { From = edge.To, To = edge.From, Value = edge.Value });
            }
            public void AddBoth(int u, int v, TEdge value = default) => AddBoth(new Edge<TEdge> { From = u, To = v, Value = value });
            private void Expand()
            {
                var temp = new Node<TEdge>[maxsize *= 2].Select(_ => _ = new Node<TEdge>()).ToArray();
                Array.Copy(G, temp, G.Length);
                G = temp;
            }
            public IEnumerable<Edge<TEdge>> GetEdges()
            {
                foreach (var node in G)
                    foreach (var edge in node)
                        yield return edge;
            }
            public int Length => G.Length;
            public Node<TEdge> this[int i] => G[i];
            public IEnumerator<Node<TEdge>> GetEnumerator() => G.ToList().GetEnumerator();
        }
        class Graph : Graph<int>
        {
            public Graph(int size = 1024) : base(size) { }
        }
        public class Node<Tedge>
        {
            public List<Edge<Tedge>> edges = new List<Edge<Tedge>>();
            public static implicit operator List<Edge<Tedge>>(Node<Tedge> node) => node.edges;
            public IEnumerator<Edge<Tedge>> GetEnumerator() => edges.GetEnumerator();
            public IEnumerable<int> destinations => edges.Select(_ => _.To);
        }
        public struct Edge<T>
        {
            public int From, To;
            public T Value;
            public Edge(int from, int to, T value = default)
            {
                From = from; To = to; Value = value;
            }
            public Edge<T> Reversed() => new Edge<T> { From = To, To = From, Value = Value };
        }


        /// <summary>
        /// ダイクストラ  始点から各頂点までの最小コストを求めます。 (有向グラフの拡張メソッド)
        /// 必要なライブラリ : [優先キュー] [グラフ]
        /// ※注意  負辺が含まれるとO(2^n)になるケースがあります。
        /// Search : O(ElogV)
        /// </summary>
        static class Dijkstraa
        {
            public static long dijkstra(this Graph<long> G, int from, int to) => dijkstra(G, from)[to];
            public static long dijkstra(this Graph<long> G, int from, int to, Comparison<(int to, long cost)> comp) => dijkstra(G, from, comp)[to];
            public static long[] dijkstra(this Graph<long> G, int from) => dijkstra(G, from, (x, y) => x.cost.CompareTo(y.cost));
            public static long[] dijkstra(this Graph<long> G, int from, Comparison<(int to, long cost)> comp)
            {
                var d = Enumerable.Repeat(long.MaxValue / 4, G.Length).ToArray();
                var que = new PriorityQueue<(int to, long cost)>(comp);
                d[from] = 0;
                que.Enqueue((from, 0));
                while (que.Count > 0)
                {
                    var (v, c) = que.Dequeue();
                    if (d[v] < c) continue;
                    foreach (var edge in G[v])
                    {
                        long ecost = (long)(edge.Value);

                        int nv = edge.To;
                        long nc = d[v] + ecost;
                        if (d[nv] > nc)
                        {
                            que.Enqueue((nv, d[nv] = nc));
                        }
                    }
                }
                return d;
            }
            static long ChLong(object value) => (long)Convert.ChangeType(value, typeof(long));
        }

        

        



        /// <summary>
        /// 無向グラフから最小全域木を作成します。　(無向グラフの拡張メソッド)
        /// 最小全域木の辺のコスト和を求めます。
        /// 必要なライブラリ : [UnionFind] [グラフ]
        /// </summary>
        static class Kruskal
        {
            /// <summary>グラフを最小全域木に変換します。</summary>
            public static Graph<long> MinimumSpanningTree(this Graph<long> G, Comparison<Edge<long>> comp = null)
            {
                comp ??= (a, b) => a.Value.CompareTo(b.Value);
                var res = new Graph<long>();
                var union = new UnionFind(G.Length);
                var edges = G.GetEdges().ToArray();
                Array.Sort(edges, comp);
                foreach (var e in edges)
                {
                    if (union.IsSame(e.From, e.To)) continue;
                    union.Unite(e.From, e.To);
                    res.AddBoth(new Edge<long> { From = e.From, To = e.To, Value = e.Value });
                }
                return res;
            }
            /// <summary>全区間の距離(コスト)の和を求めます。</summary>
            public static long CostSum(this Graph<long> G)
            {
                var res = 0L;
                foreach (var item in G.GetEdges())
                {
                    res += item.Value;
                }
                return res / 2;
            }
        }


        /// <summary>
        /// トポロジカルソート  DAGを順序付けし、起点の要素から並べた一次元配列に直します。
        /// </summary>
        static class TopologicalSort
        {
            // TODO Degreesの実装
            /// <summary>トポロジカルソートした結果を返します。不可能な場合nullを返します。</summary>
            /// <param name="G">グラフ</param>
            /// <param name="Degrees">Degreesはその頂点にのびる辺の数</param>
            /// <returns></returns>
            public static List<int> Topologicalsort(Graph<int> G, int[] Degrees)
            {
                var que = new Queue<int>();
                for (int i = 0; i < Degrees.Length; i++)
                {
                    if (Degrees[i] == 0) que.Enqueue(i);
                }
                var sorted = new List<int>();
                while (que.Count > 0)
                {
                    var v = que.Dequeue();
                    sorted.Add(v);
                    foreach (var i in G[v])
                    {
                        Degrees[i.To]--;
                        if (Degrees[i.To] == 0) que.Enqueue(i.To);
                    }
                }
                if (Degrees.Length != sorted.Count) return null;
                return sorted;
            }
        }
    }

        



        
    namespace Others
    {
        //TODO 区間加算の実装 このままではセグ木で物足りてしまう
        /// <summary>
        /// BIT  セグ木より定数倍高速な範囲加算に。
        /// Sum : O(log(N))
        /// Add : O(log(N))
        /// </summary>
        class BIT
        {
            int N;
            long[] data;
            public BIT(int size)
            {
                data = new long[size + 1];
                N = size;
            }

            public long Sum(int i)//[0,i)
            {
                long s = 0;
                while (i > 0)
                {
                    s += data[i];
                    i -= (i & -i);
                }
                return s;
            }
            public void Add(int i, long x)
            {
                i++;
                while (i <= N)
                {
                    data[i] += x;
                    i += (i & (-i));
                }
            }
            public IEnumerator GetEnumerator()
            {
                for (int i = 0; i < N; i++)
                {
                    yield return this[i];
                }
            }
            public long this[int i]
            {
                set => Add(i, value - this[i]);
                get => Sum(i + 1) - Sum(i);
            }
        }

        /// <summary>
        /// 二次元BIT
        /// 単更新 & 二次元区間和の取得 Log(N)
        /// </summary>
        class BIT2D
        {
            public int X, Y;
            int[,] bit;
            public BIT2D(int X, int Y)
            {
                this.X = X;
                this.Y = Y;
                bit = new int[X + 1, Y + 1];
            }
            public BIT2D(int[,] array) : this(array.GetLength(0), array.GetLength(1))
            {
                for (int i = 0; i < X; i++)
                    for (int j = 0; j < Y; j++)
                        Add(i, j, array[i, j]);
            }
            public void Add(int x, int y, int value)
            {
                for (int i = x + 1; i <= X; i += i & (-i))
                    for (int j = y + 1; j <= Y; j += j & (-j))
                        bit[i, j] += value;
            }
            public int Sum(int x0, int y0, int x1, int y1)
            {
                return Sum(x0, y0) + Sum(x1, y1) - Sum(x0, y1) - Sum(x1, y0);
            }
            int Sum(int x, int y)
            {
                var sum = 0;
                for (var i = x; 0 < i; i -= i & (-i))
                    for (var j = y; 0 < j; j -= j & (-j))
                        sum += bit[i, j];
                return sum;
            }
        }

        // Zアルゴリズム 先頭文字列と何文字一致しているか
        class ZAlgorithm
        {
            static string S;
            int[] Same;
            public ZAlgorithm(string s)
            {
                S = s;
                Same = Search();
            }
            static int[] Search()
            {
                int N = S.Length, c = 0;
                var Z = new int[N];
                for (int i = 1; i < N; i++)
                {
                    int l = i - c;
                    if (i + Z[l] < c + Z[c])
                    {
                        Z[i] = Z[l];
                    }
                    else
                    {
                        int leng = Math.Max(0, c + Z[c] - i);
                        while (leng + i < N && S[i + leng] == S[leng])
                        {
                            leng++;
                        }
                        Z[i] = leng;
                        c++;
                    }
                }
                Z[0] = N;
                return Z;
            }
            public int this[int i] => Same[i];
        }



        /// <summary>
        /// defaultdict  pythonのdefaultdiftを参考にした。
        /// 無いキーを参照するとdefaultの値を返す。
        /// </summary>
        class DefaultDict<TKey, TValue> : Dictionary<TKey, TValue> //Dictionary
        {
            new public TValue this[TKey key]
            {
                set => base[key] = value;
                get => TryGetValue(key, out TValue value) ? value : default;
            }
        }


        /// <summary>
        /// 入力に番号を割り振ります
        /// </summary>
        /// <typeparam name="T">入力の型</typeparam>
        class IndexConverter<T> : IEnumerable<T>
        {
            Dictionary<T, int> itemToIndex = new Dictionary<T, int>();
            List<T> indexToItem = new List<T>();
            public int Add(T item)
            {
                if (itemToIndex.ContainsKey(item)) return itemToIndex[item];
                indexToItem.Add(item);
                return itemToIndex[item] = itemToIndex.Count;
            }
            public int IndexOf(T item) => itemToIndex[item];
            public IEnumerator<T> GetEnumerator() => indexToItem.GetEnumerator();
            IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
            public int this[T item] => itemToIndex[item];
            public T this[int index] => indexToItem[index];
            public int Count => itemToIndex.Count;
        }


        /// <summary>
        /// 入力を独立な値に直し、ソートして番号を割り振ります。
        /// データの追加は初期化時のみできます。
        /// 必要なライブラリ : MySortedSet<T>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class SortedIndexConverter<T> where T : IComparable
        {
            Dictionary<T, int> itemToIndex = new Dictionary<T, int>();
            List<T> indexToItem = new List<T>();
            public SortedIndexConverter(IEnumerable<T> items)
            {
                var sorted = new SortedSet<T>(items);
                foreach (var item in sorted)
                {
                    itemToIndex.Add(item, itemToIndex.Count);
                    indexToItem.Add(item);
                }
            }
            public SortedIndexConverter(IEnumerable<T> items, Comparison<T> comp)
            {
                var sorted = new MySortedSet<T>(items, comp);
                foreach (var item in sorted)
                {
                    itemToIndex.Add(item, itemToIndex.Count);
                    indexToItem.Add(item);
                }
            }
            public int this[T item] => itemToIndex[item];

            public int Count => itemToIndex.Count;
        }

        /// <summary>
        /// ComparisonをラップしたSortedSet
        /// Comparisonでの比較ができる。
        /// (int型でのテスト済み)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        class MySortedSet<T> : SortedSet<T> where T : IComparable
        {
            public MySortedSet() : base() { }
            public MySortedSet(IEnumerable<T> data) : base(data) { }
            public MySortedSet(Comparison<T> comp) : base(new Comp<T>(comp)) { }
            public MySortedSet(IEnumerable<T> data, Comparison<T> comp) : base(data, new Comp<T>(comp)) { }
            /// <summary>Comparisonのラッパー</summary>
            class Comp<U> : IComparer<U> where U : IComparable
            {
                Comparison<U> comp;
                public Comp(Comparison<U> _comp) => comp = _comp;
                public int Compare(U x, U y) => comp(x, y);
            }
        }
    }


        



    
    







        


        

        
        




        /// <summary>
        /// 累積和  ちょっと使いにくい
        /// [a,b)の区間和の取得 Log(N)
        /// </summary>
        public class CumulativeSum
        {
            long[] D, A;
            bool IsCalcd = false;
            Func<long, long, long> func;
            public CumulativeSum(int[] A) : this(A, (i, j) => i + j) { }
            public CumulativeSum(int size, Func<long, long, long> func) : this(new int[size], func) { }
            public CumulativeSum(int size) : this(new int[size]) { }
            public CumulativeSum(int[] A, Func<long, long, long> func)
            {
                D = new long[A.Length + 1];
                Array.Copy(A, this.A = new long[A.Length], A.Length);
                this.func = func;
            }
            private void Calc()
            {
                if (IsCalcd) return;
                for (int i = 0; i < A.Length; i++) D[i + 1] = func(D[i], A[i]);
                IsCalcd = true;
            }
            public long this[int i] { get { Calc(); return D[i + 1] - D[i]; } set { A[i] = (int)value; IsCalcd = false; } }
            public long this[int from, int to] { get { Calc(); return D[to] - D[from]; } }
            public int Length => D.Length - 1;
        }




        /// <summary>
        /// 二次元累積和  二次元区間の区間和を求めます。
        /// 
        /// </summary>
        class CumulativeSum2D
        {
            long[,] D;
            int H, W;
            /// <summary>O(HW)</summary>
            public CumulativeSum2D(long[,] A)
            {
                H = A.GetLength(0);
                W = A.GetLength(1);
                D = new long[H + 1, W + 1];
                Copy(A);
                Culc();
            }
            void Culc()
            {

                for (int i = 0; i < H; i++)
                {
                    for (int j = 0; j < W + 1; j++)
                    {
                        D[i + 1, j] += D[i, j];
                    }
                }
                for (int i = 0; i < H + 1; i++)
                {
                    for (int j = 0; j < W; j++)
                    {
                        D[i, j + 1] += D[i, j];
                    }
                }
            }
            void Copy(long[,] A)
            {
                for (int i = 0; i < H; i++)
                {
                    for (int j = 0; j < W; j++)
                    {
                        D[i + 1, j + 1] = A[i, j];
                    }
                }
            }
            /// <summary>
            /// y : [0,H]
            /// x : [0,W]
            /// を指定して指定した範囲の区間和を求める。
            /// </summary>
            public long GetSum(int y1, int x1, int y2, int x2)
            {
                if (y2 < y1) (y1, y2) = (y2, y1);
                if (x2 < x1) (x2, x1) = (x1, x2);
                var sum = D[y2, x2] + D[y1, x1] - D[y2, x1] - D[y1, x2];
                return sum;
            }
            public long this[int i, int j] => D[i, j];
        }




        static class LCS
        {
            public static int Search(int[] aList, int[] blist)
            {
                int n = aList.Length;
                int m = blist.Length;
                var dp = new int[n + 1][];
                for (int i = 0; i <= n; i++)
                {
                    dp[i] = Enumerable.Repeat(int.MaxValue, m + 1).ToArray();
                }
                dp[0][0] = 0;
                for (int i = 0; i < n + 1; i++)
                {
                    for (int j = 0; j < m + 1; j++)
                    {
                        if (i > 0)
                        {
                            dp[i][j] = Math.Min(dp[i][j], dp[i - 1][j] + 1);
                        }
                        if (j > 0)
                        {
                            dp[i][j] = Math.Min(dp[i][j], dp[i][j - 1] + 1);
                        }
                        if (i > 0 && j > 0)
                        {
                            dp[i][j] = Math.Min(dp[i][j], dp[i - 1][j - 1] + (aList[i - 1] == blist[j - 1] ? 0 : 1));
                        }
                    }
                }
                return dp[n][m];
            }
        }




        

        /// <summary>
        /// 座標圧縮
        /// </summary>
        static class CoordinateCompression
        {
            public static int[] Compress(int[] A)
            {
                var nA = A.OrderBy(_ => _).Distinct().ToArray();
                var N = A.Length;
                var nN = nA.Length;
                var ans = new List<int>(N);

                foreach (var a in A)
                {
                    var (lb, ub) = (-1, nN);
                    while (ub - lb > 1)
                    {
                        var mid = (lb + ub) / 2;
                        (lb, ub) = nA[mid] <= a ? (mid, ub) : (lb, mid);
                    }
                    ans.Add(lb);
                }
                return ans.ToArray();
            }
        }

    



    



    // 半分全列挙 個数40のナップザック問題
    static class HalfFullEnumeration
    {
        //二分探索の部分が0(logN)以下の処理ならなんでもいい
        //[例]ハッシュ検索
        public static long Knapsack(long[] v, long[] w, long W)
        {
            int N = v.Length;
            var (n1, n2) = (N / 2, N - N / 2);
            var A1 = new (long v, long w)[n1];
            var A2 = new (long v, long w)[n2];
            {
                int index = 0;
                while (index < n1) A1[index] = (v[index], w[index++]);
                while (index < N) A2[index - n1] = (v[index], w[index++]);
            }
            var P1 = CompressedEnum(A1, n1);
            var P2 = CompressedEnum(A2, n2);
            long res = 0;
            for (int i = 0; i < P2.Length; i++)
            {
                var (lb, ub) = (-1, P1.Length);
                while (ub - lb > 1)
                {
                    int mid = (ub + lb) / 2;
                    if (P2[i].w + P1[mid].w <= W) lb = mid;
                    else ub = mid;
                }
                if (lb == -1) continue;
                res = Math.Max(res, P2[i].v + P1[lb].v);
            }
            return res;
        }
        private static (long v, long w)[] CompressedEnum((long v, long w)[] A, int size)
        {
            var enm = new (long v, long w)[1 << size];
            for (int i = 0; i < (1 << size); i++)
            {
                var (sv, sw) = (0L, 0L);
                for (int j = 0; j < size; j++)
                {
                    if ((i & (1 << j)) >= 1)
                    {
                        sv += A[j].v;
                        sw += A[j].w;
                    }
                }
                enm[i] = (sv, sw);
            }
            //重み順にソート
            Array.Sort(enm, (x, y) => x.w.CompareTo(y.w));
            //価値順にソート(邪魔するものは、消去)
            var res = new List<(long, long)>();
            long maxv = -1;
            for (int i = 0; i < enm.Length; i++)
            {
                if (enm[i].v <= maxv) continue;
                maxv = enm[i].v;
                res.Add(enm[i]);
            }
            return res.ToArray();
        }
    }



    

    



    // 遅延評価セグメントツリー
    /// <summary>
    /// qfunc: (左,右) クエリ時のデータのマージ
    /// updfunc: (左,右) アプデ時のデータのマージ
    /// efunc1: (元Lazyの値,範囲の差(2の倍数),記録する値) Lazyに移す
    /// efunc2: (子Lazyの値,親Lazyの値) 親から子Lazyに伝搬する
    /// efunc3: (元データの値,Lazyの値) Lazyをデータに移す
    ///
    /// StarrySkyTreeの入力例 (new int[N],(x,y)=>Max(x,y),(x,y)=>Max(x,y),(x,y,z)=>x+z,(x,y)=>x+y,(x,y)=>x+y)
    /// </summary
    /*
    class SegTree2D<T>
    {
        int H, W;
        T init;
        T[,] dat;
        Func<T, T, T> updFunc, qFunc;
        public SegTree2D(int h,int w, T _init, Func<T, T, T> updfunc, Func<T, T, T> qfunc)
        {
            H = W = 1;
            while (H < h) H <<= 1;
            while (W < w) W <<= 1;
            dat = new T[2 * H - 1, 2 * W - 1];
            dat.Fill(_init);
            updFunc = updfunc;
            qFunc = qfunc;
            init = _init;
        }
        public void Update(int h,int w, T x)
        {
            a += N - 1;
            dat[a] = x;
            while (a > 0)
            {
                a = (a - 1) >> 1;
                dat[a] = updFunc(dat[(a << 1) + 1], dat[(a << 1) + 2]);
            }
        }
        public T Query(int a) => dat[N - 1 + a]; //点aを検索 O(1)
        public T Query(int a, int b) => Query(a, b, 0, 0, N); //[a, b)を検索 O(logN)
        private T Query(int a, int b, int k = 0, int l = 0, int r = 0)
        {
            if (r <= a || b <= l) return init;
            else if (a <= l && r <= b) return dat[k];
            else
            {
                T vl = Query(a, b, (k << 1) + 1, l, (l + r) >> 1);
                T vr = Query(a, b, (k << 1) + 2, (l + r) >> 1, r);
                return qFunc(vl, vr);
            }
        }
        public T this[int a] { get => Query(a); set => Update(a, value); }
        public T this[int a, int b] => Query(a, b);
    }
    */


    //抽象化遅延評価セグメントツリー [a,b)に対する更新・取得
    class LazySegTree<T> where T : struct, IComparable
    {
        int n;
        T[] Data, Lazy;
        T defalt = default;
        Func<T, T, T> qfunc, updfunc, efunc2, efunc3;
        Func<T, int, T, T> efunc1;
        public LazySegTree(T[] v, T init, Func<T, T, T> _qfunc, Func<T, T, T> _updfunc, Func<T, int, T, T> _efunc1, Func<T, T, T> _efunc2, Func<T, T, T> _efunc3)
        {
            int size = v.Length;
            qfunc = _qfunc;
            updfunc = _updfunc;
            efunc1 = _efunc1;
            efunc2 = _efunc2;
            efunc3 = _efunc3;
            defalt = init;
            n = 1; while (n < size) n <<= 1;
            Data = new T[n * 2 - 1];
            Lazy = new T[n * 2 - 1];
            for (int i = 0; i < size; i++) Data[i + n - 1] = v[i];
            for (int i = n - 2; i >= 0; i--) Data[i] = updfunc(Data[i * 2 + 1], Data[i * 2 + 2]);
        }
        public LazySegTree(int size, T init, Func<T, T, T> _qfunc, Func<T, T, T> _updfunc, Func<T, int, T, T> _efunc1, Func<T, T, T> _efunc2, Func<T, T, T> _efunc3) : this(new T[size], init, _qfunc, _updfunc, _efunc1, _efunc2, _efunc3) { }

        void eval(int k, int l, int r)//nodeが呼び出された時に伝達する。
        {
            if (Lazy[k].CompareTo(defalt) != 0)
            {
                Data[k] = efunc3(Data[k], Lazy[k]); // Data[k]+Lazy[k]
                if (r - l > 1)
                {
                    Lazy[k * 2 + 1] = efunc2(Lazy[k * 2 + 1], Lazy[k]); // Lazy[k]/2
                    Lazy[k * 2 + 2] = efunc2(Lazy[k * 2 + 2], Lazy[k]); // Lazy[k]/2
                }
            }
            Lazy[k] = default;
        }
        public void Update(int a, T x) => Update(a, a + 1, x, 0, 0, n);
        public void Update(int a, int b, T x) => Update(a, b, x, 0, 0, n);
        private void Update(int a, int b, T x, int k, int l, int r)
        {
            eval(k, l, r);
            if (b <= l || r <= a) return; // 範囲外
            if (a <= l && r <= b)// 範囲内
            {
                Lazy[k] = efunc1(Lazy[k], r - l, x); // Lazy[k]+(r-l)*x
                eval(k, l, r);
            }
            else// 片方だけ範囲外
            {
                Update(a, b, x, k * 2 + 1, l, (l + r) >> 1);
                Update(a, b, x, k * 2 + 2, (l + r) >> 1, r);
                Data[k] = updfunc(Data[k * 2 + 1], Data[k * 2 + 2]);
            }
        }
        public T Query(int a, int b) => Query(a, b, 0, 0, n);
        public T Query(int a) => Query(a, a + 1, 0, 0, n);
        private T Query(int a, int b, int k, int l, int r)
        {
            if (b <= l || r <= a) return default;
            eval(k, l, r);
            if (a <= l && r <= b) return Data[k];
            else
            {
                var vl = Query(a, b, k * 2 + 1, l, (l + r) >> 1);
                var vr = Query(a, b, k * 2 + 2, (l + r) >> 1, r);
                return qfunc(vl, vr);
            }
        }
        public T this[int a] { get => Query(a); set => Update(a, value); }
        public T this[int a, int b] { get => Query(a, b); set => Update(a, b, value); }
    }

    
   


    // ビットDPのサンプル (徘徊セールスマン問題)
    // bitをテーブルにのせてDPするだけ(メモ化再帰)
    // n:地点数    d:距離行列
    // S:訪れた地点 v:現在地点

    //static int rec(int S, int v)
    //{
    //    if (dp[S, v] >= 0) return dp[S, v]; //更新されている=この地点からの最小は求まっている
    //    if (S == (1 << n) - 1 && v == 0) return dp[S, v] = 0;
    //    int res = int.MaxValue / 4;
    //    for (int u = 0; u < n; u++)
    //    {
    //        if ((S & (1 << u)) == 0)
    //        {
    //            res = Min(res, rec(S | (1 << u), u) + d[v, u]);
    //        }
    //    }
    //    return dp[S, v] = res;
    //}

    

    
    





    namespace TEMPLATES
    {
        public static class StaticScanner
        {
            static System.IO.Stream reader;
            static byte[] buffer = new byte[1024];
            static int cursor = 0, length = 0;
            static StaticScanner()
            {

                reader = Console.OpenStandardInput();
            }
            public static void CloseReader() => reader.Close();
            public static string RStr()
            {
                var line = new System.Text.StringBuilder();
                char c;
                while (true)
                {
                    if (cursor == length)
                    {
                        length = reader.Read(buffer);
                        cursor = 0;
                        if (length == 0) break;
                    }
                    c = (char)buffer[cursor++];
                    if (c == '\n') break;
                    line.Append(c);
                }
                return line.ToString();
            }
            public static int RInt() => RTuple<int>();
            public static long RLong() => RTuple<long>();
            public static double RDouble() => RTuple<double>();
            public static string[] RStrs() => RStr().Split();
            public static int[] RInts() => Array.ConvertAll(RStrs(), int.Parse);
            public static long[] RLongs() => Array.ConvertAll(RStrs(), long.Parse);
            public static double[] RDoubles() => Array.ConvertAll(RStrs(), double.Parse);
            public static int[] RInts(Func<int, int> func) => RInts().Select(func).ToArray();
            public static long[] RLongs(Func<long, long> func) => RLongs().Select(func).ToArray();
            public static double[] RDoubles(Func<double, double> func) => RDoubles().Select(func).ToArray();
            private static T ChType<T>(string r) => (T)Convert.ChangeType(r, typeof(T));
            public static T1 RTuple<T1>()
            {
                var r = RStr();
                return ChType<T1>(r);
            }
            public static (T1, T2) RTuple<T1, T2>()
            {
                var r = RStrs();
                return (ChType<T1>(r[0]), ChType<T2>(r[1]));
            }
            public static (T1, T2, T3) RTuple<T1, T2, T3>()
            {
                var r = RStrs();
                return (ChType<T1>(r[0]), ChType<T2>(r[1]), ChType<T3>(r[2]));
            }
            public static (T1, T2, T3, T4) RTuple<T1, T2, T3, T4>()
            {
                var r = RStrs();
                return (ChType<T1>(r[0]), ChType<T2>(r[1]), ChType<T3>(r[2]), ChType<T4>(r[3]));
            }
            public static (T1, T2, T3, T4, T5) RTuple<T1, T2, T3, T4, T5>()
            {
                var r = RStrs();
                return (ChType<T1>(r[0]), ChType<T2>(r[1]), ChType<T3>(r[2]), ChType<T4>(r[3]), ChType<T5>(r[4]));
            }
            //N行入力 => (T1,T2..)[N]
            public static (T1, T2)[] RTuples<T1, T2>(int N) => new (T1, T2)[N].Select(_ => _ = RTuple<T1, T2>()).ToArray();
            public static (T1, T2, T3)[] RTuples<T1, T2, T3>(int N) => new (T1, T2, T3)[N].Select(_ => _ = RTuple<T1, T2, T3>()).ToArray();
            public static (T1, T2, T3, T4)[] RTuples<T1, T2, T3, T4>(int N) => new (T1, T2, T3, T4)[N].Select(_ => _ = RTuple<T1, T2, T3, T4>()).ToArray();
        }


        public static class Template
        {
            public static T[] Copy<T>(this T[] sourse)
            {
                T[] res = new T[sourse.Length];
                Array.Copy(sourse, res, sourse.Length);
                return res;
            }
            public static void CopyTo<T>(this T[] source, T[] destination) => Array.Copy(source, destination, source.Length);
            public static void CopyTo(this Array ary, Array dest) => Array.Copy(ary, dest, ary.Length);
            public static void Fill<T>(this T[] ary, T init) => ary.AsSpan().Fill(init);
            public static void Fill<T>(this T[,] ary, T init) => MemoryMarshal.CreateSpan(ref ary[0, 0], ary.Length).Fill(init);
            public static void Fill<T>(this T[,,] ary, T init) => MemoryMarshal.CreateSpan(ref ary[0, 0, 0], ary.Length).Fill(init);
            public static void Sort<T>(this T[] ary) where T : IComparable<T>
            => Array.Sort(ary);
            public static void Sort<T>(this T[] ary, Comparison<T> comp) where T : IComparable<T> => Array.Sort(ary, comp);
            public static void Sort<T>(this T[] sourse, params T[][] dest) where T : IComparable<T>
            {
                var E = Enumerable.Range(0, sourse.Length).ToArray();
                Array.Sort(sourse, E);
                foreach (Span<T> item in dest)
                {
                    Span<T> cd = new T[item.Length];
                    item.CopyTo(cd);
                    for (int j = 0; j < sourse.Length; j++) item[j] = cd[E[j]];
                }
            }
            public static void Reverse<T>(this T[] ary) where T : IComparable<T> => Array.Reverse(ary);
            public static string JoinedString<T>(this IEnumerable<T> ary, string sep = " ") =>
            string.Join(sep, ary);
            public static void WriteLine<T>(T str) { Console.WriteLine(str.ToString()); }
            public static bool ChMax<T>(ref this T value, T other) where T : struct, IComparable<T>
            {
                if (value.CompareTo(other) < 0)
                {
                    value = other;
                    return true;
                }
                return false;
            }
            public static bool ChMin<T>(ref this T value, T other) where T : struct, IComparable<T>
            {
                if (value.CompareTo(other) > 0)
                {
                    value = other;
                    return true;
                }
                return false;
            }
            public static void Swap<T>(ref this (T, T) item) =>
        item = (item.Item2, item.Item1);
            //[min,max]の範囲内かどうか
            public static bool IsBetween<T>(ref this T value, T min, T max) where T : struct, IComparable<T>
            {
                return 0 <= value.CompareTo(min) && value.CompareTo(max) < 0;
            }
            public static string YESNO(bool b) => b ? "YES" : "NO";
            public static string YesNo(bool b) => b ? "Yes" : "No";
            public static string yesno(bool b) => b ? "yes" : "no";
            public static readonly int Inf = int.MaxValue / 2;
            public static readonly long InfL = long.MaxValue / 4;
            public static readonly (int, int)[] DD4 = new (int y, int x)[4] { (1, 0), (0, 1), (-1, 0), (0, -1) };
            public static readonly (int, int)[] DD8 = new (int y, int x)[8] { (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1) };
        }


        /// <summary>
        /// 
        /// </summary>
        namespace Debug
        {
            /// <summary>
            /// 入力ファイルから読み込み、出力ファイルに出力する。
            /// 実行ディレクトリにinput.txtとoutput.txtを置くこと
            /// インスタンス名にConsoleとつけることで入力の置き換えが可能
            /// </summary>
            public class FileIO
            {
                string INPUTPATH, OUTPUTPATH;
                string[] alllines;
                public FileIO(string inpath = @"input.txt", string outpath = @"output.txt")
                {
                    INPUTPATH = inpath;
                    OUTPUTPATH = outpath;
                    alllines = File.ReadAllLines(inpath);
                    Clear();
                }
                int currentline = 0;
                public string ReadLine()
                {
                    if (alllines.Length <= currentline) throw new Exception("ファイルの終わりに到達しました。");
                    return alllines[currentline++];
                }
                public string[] ReadAllLines()
                {
                    string[] copy = new string[alllines.Length];
                    Array.Copy(alllines, copy, alllines.Length);
                    return alllines;
                }
                public void WriteLine(string str)
                {
                    File.AppendAllText(OUTPUTPATH, str + Environment.NewLine);
                }
                public void Clear()
                {
                    File.WriteAllText(OUTPUTPATH, "");
                }
                public string GetOutputPath => Directory.GetCurrentDirectory() + "/" + OUTPUTPATH;
                public string GetInput => Directory.GetCurrentDirectory() + "/" + INPUTPATH;
            }
        }

        static class RandomInput
        {
            //"using staticを置き換えることで入力をランダムにする" ことが目標
            public static Random Rand = new Random(DateTime.Now.Millisecond);
            public static int RandomInt(int min, int max) => Rand.Next(min, max);
            public static string RandomString(int min = 4, int max = 8) => string.Join("", new char[RandomInt(min, max)].Select(_ => (char)('a' + RandomInt(0, 26))).ToArray());
            public static int[] RandomArray(int leng, int minvalue, int maxvalue) => new int[leng].Select(_ => Rand.Next(minvalue, maxvalue)).ToArray();
            /// <summary>ランダムなbool[H,W]を作る</summary>
            static bool[,] randommap(int H, int W, float freq)
            {
                Random rand = new Random();
                var res = new bool[H, W];
                for (int i = 0; i < H; i++)
                {
                    for (int j = 0; j < W; j++)
                    {
                        res[i, j] = rand.NextDouble() <= freq;
                    }
                }
                return res;
            }
            public static int RInt() => RandomInt(1, 10);
            public static int[] RInts() => RandomArray(10, 0, 10);
            //var sw = new System.IO.StreamWriter(Console.OpenStandardOutput()) { AutoFlush = false };
            //Console.SetOut(sw);
            //Console.Out.Flush();
        }
    }
}



/// <summary>
/// ヒューリスティックコンテストのためのデータ構造
/// </summary>
namespace ForHeuristic
{
    /// <summary>
    /// bool二次配列の代わり
    /// メモリ使用量がとても少ない
    /// </summary>
    class BitArray2D
    {
        ulong[,] data;
        const ulong all = 0xffffffffffffffff;
        public BitArray2D(int sizey, int sizex)
        {
            long size = sizey * sizex;
            int x = sizex / 64 + 1;
            int y = sizey;
            data = new ulong[y, x];
        }
        public void SetRange(int fy, int fx, int ty, int tx, bool b = true)
        {
            for (int y = fy; y <= ty; y++)
            {
                int nx;
                for (int x = fx; x < tx + 64 - (tx & 0x3F); x = nx)
                {
                    nx = x + 64 - (x & 0x3F);
                    ulong mask = all;
                    mask &= all >> (x & 0x3F);
                    if (tx < nx) mask &= all << (63 - (tx & 0x3F));
                    if (b) data[y, x / 64] |= mask;
                    else data[y, x / 64] &= ~mask;
                }
            }
        }
        public bool GetRangeOR(int fy, int fx, int ty, int tx)
        {
            bool ans = false;
            for (int y = fy; y <= ty; y++)
            {
                int nx;
                for (int x = fx; x < tx + 64 - (tx & 0x3F); x = nx)
                {
                    nx = x + 64 - (x & 0x3F);
                    ulong mask = all;
                    mask &= all >> (x & 0x3F);
                    if (tx < nx) mask &= all << (63 - (tx & 0x3F));
                    ans |= (data[y, x / 64] & mask) > 0;
                    if (ans) return true;
                }
            }
            return false;
        }
        public bool this[int y, int x]
        {
            get
            {
                return (data[y, x / 64] & (1UL << 63) >> (x & 0x3F)) > 0;
            }
            set
            {
                if (value) data[y, x / 64] |= 1UL << (63 - (x & 0x3F));
                else data[y, x / 64] ^= (1UL << (63 - (x & 0x3F)));
            }
        }
    }
}








//////////////////////////////////////////////////////////////////////////////

/// //////////////////////////////////////////////////////////////////////////
///                                                                        ///
///                      ここから開発途中のライブラリ                           ///
///                                                                        ///
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////






namespace TEST
{
    using CHANGEABLEGRAPH;
    /// <summary>
    /// グラフにDictinaryを乗せてみた
    /// 辺の追加が通常の2倍遅い(20000辺 => 5.74ms)
    /// ダイクストラの速度は大して変わらなかった (100ダイク*10000 => 189ms)
    /// 最大流に対応するためEdgeはクラスにしている。
    /// 辺重複不可 辺削除可
    /// 
    /// ( Nodeはforeachで回した時にedgeを取り出す動作とインデックスによるsetを両立するためにある。)
    /// </summary>
    namespace CHANGEABLEGRAPH
    {
        class Graph<TEdge>
        {
            int maxsize = 0;
            bool arrowexpand = true;
            private Node<TEdge>[] G;
            public Graph(int size = 4)
            {
                arrowexpand = size == 4;
                maxsize = size;
                G = new Node<TEdge>[size].Select(_ => _ = new Node<TEdge>()).ToArray();
            }
            public void Add(Edge<TEdge> edge)
            {
                while (Math.Max(edge.From, edge.To) >= maxsize) Expand();
                G[edge.From].edges[edge.To] = edge;
            }
            public void Add(int from, int to, TEdge val = default) => Add(new Edge<TEdge>(from,to,val));
            public void Remove(Edge<TEdge> edge) => G[edge.From].edges.Remove(edge.To);
            public void Remove(int from, int to, TEdge val = default) => Remove(new Edge<TEdge>(from, to, val));
            public void Change(Edge<TEdge> edge) => Add(edge);
            public bool Contains(Edge<TEdge> edge) => G[edge.From].edges.ContainsKey(edge.To);
            public bool Contains(int from, int to, TEdge val = default) => Contains(new Edge<TEdge>(from, to, val));
            public void AddBoth(Edge<TEdge> edge)
            {
                Add(edge);
                Add(new Edge<TEdge> { From = edge.To, To = edge.From, Value = edge.Value });
            }
            public void AddBoth(int from, int to, TEdge val = default) => AddBoth(new Edge<TEdge>(from, to, val));
            public IEnumerable<Edge<TEdge>> GetEdges()
            {
                foreach (var node in G)
                    foreach (var edge in node)
                        yield return edge;
            }
            private void Expand()
            {
                if (!arrowexpand) throw new Exception("グラフが指定したサイズを超える入力を受け取りました。");
                var temp = new Node<TEdge>[maxsize *= 2].Select(_ => _ = new Node<TEdge>()).ToArray();
                Array.Copy(G, temp, G.Length);
                G = temp;
            }
            public int Length => G.Length;
            public Node<TEdge> this[int i] => G[i];
            public IEnumerator<Node<TEdge>> GetEnumerator() => G.ToList().GetEnumerator();
        }
        class Graph : Graph<int> { public Graph(int size = 4) : base(size) { } }
        public class Node<Tedge>
        {
            public Dictionary<int, Edge<Tedge>> edges = new Dictionary<int, Edge<Tedge>>();
            public IEnumerable destinations => edges.Select(_ => _.Value.To);
            public static implicit operator List<Edge<Tedge>>(Node<Tedge> node) => node.edges.Values.ToList();
            public IEnumerator<Edge<Tedge>> GetEnumerator() => edges.Values.GetEnumerator();
            public Edge<Tedge> this[int i] { get => edges.TryGetValue(i, out Edge<Tedge> edge) ? edge : default; set => edges[i] = value; }
        }
        public class Edge<T>
        {
            public Edge(int from, int to, T val = default) { From = from; To = to; Value = val; }
            public Edge() { }
            public int From, To;
            public T Value;
            public Edge<T> Reversed() => new Edge<T> { From = To, To = From, Value = Value };
        }
    }


    static class MaxFlow
    {
        static bool[] used;
        static Graph<int> G;
        public static int GetMaxFlow(this Graph<int> G, int s, int g)
        {
            int flow = 0;
            int V = G.Length;
            G = new Graph<int>(V + 1);
            foreach (var item in G.GetEdges())
            {
                if (!G.Contains(new Edge<int> { From = item.To, To = item.From }))
                {
                    G.Add(new Edge<int> { From = item.To, To = item.From, Value = 0 });
                }
            }
            while (true)
            {
                used = new bool[V];
                used.Fill(false);
                int f = dfs(s, g, int.MaxValue / 2);
                if (f == 0) break;
                flow += f;
            }
            return flow;
        }
        static int dfs(int v, int t, int f)
        {
            if (v == t) return f;
            used[v] = true;
            var edges = G[v].edges.Values.ToArray();
            foreach (var edge in edges)
            {
                var (to, from, value) = (edge.To, edge.From, edge.Value);
                if (!used[to] && value > 0)
                {
                    int d = dfs(to, t, Math.Min(f, value));
                    if (d > 0)
                    {
                        G[from][to].Value -= d;
                        G[to][from].Value += d;
                        return d;
                    }
                }
            }
            return 0;
        }
    }

    //独立な要素のみ格納されるQueue
    class IndependentQueue<T>
    {
        Queue<T> que = new Queue<T>();
        HashSet<T> hash = new HashSet<T>();
        public IndependentQueue() { }
        public void Enqueue(T item)
        {
            if (!hash.Contains(item))
            {
                que.Enqueue(item);
            }
        }
        public T Dequeue()
        {
            var item = que.Dequeue();
            hash.Remove(item);
            return item;
        }
        public T Peek()
        {
            return que.Peek();
        }
        public int Count => hash.Count;
    }



    // 両端キュー 双方向からpushできるqueue
    



}

namespace SegTree
{
    //セグ木にデータを持たせたかったがジェネリクスのせいでうまくいかなかった
    //自分自身の型でジェネリクスして、内部で型を持たせるようにした。
    /// <summary>
    /// モノイドとは
    /// (a•b)•c = a•(b•c) の性質をもつ演算子と単位元の組
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMonoid<T>
    {
        public T unit { get; }
        public T Merge(T other);
    }

    struct RangeMax : IMonoid<RangeMax>
    {
        public RangeMax unit => new(long.MinValue);
        public RangeMax Merge(RangeMax other) => new(Math.Max(this, other));

        public long Value;
        public RangeMax(long value) => Value = value;
        public static implicit operator long(RangeMax value) => value.Value;
    }

    struct RangeMin : IMonoid<RangeMin>
    {
        public RangeMin unit => new(long.MaxValue);
        public RangeMin Merge(RangeMin other) => new(Math.Min(this, other));

        public long Value;
        public RangeMin(long value) => Value = value;
        public static implicit operator long(RangeMin value) => value.Value;
    }
    /// <summary>
    /// セグメントツリー
    /// モノイド を実装した
    /// </summary>
    class SegTree<T> where T:IMonoid<T>
    {
        int N = 1;
        T[] dat;
        public SegTree(int n)
        {
            while (N <= n) N <<= 1;
            dat = new T[2 * N - 1];
        }
        public void Update(int a, T x)
        {
            a += N - 1;
            dat[a] = x;
            while (a > 0)
            {
                a = (a - 1) >> 1;
                dat[a] = dat[(a << 1) + 1].Merge(dat[(a << 1) + 2]);
            }
        }
        /// <summary>点aを検索 O(1)</summary>
        public T Query(int a) => dat[N - 1 + a];
        ///<summary>[a, b)を検索 O(logN)</summary>
        public T Query(int a, int b) => Query(a, b, 0, 0, N);
        T Query(int a, int b, int k = 0, int l = 0, int r = 0)
        {
            if (r <= a || b <= l) return default(T).unit;
            else if (a <= l && r <= b) return dat[k];
            else
            {
                T vl = Query(a, b, (k << 1) + 1, l, (l + r) >> 1);
                T vr = Query(a, b, (k << 1) + 2, (l + r) >> 1, r);
                return vl.Merge(vr);
            }
        }
        public T this[int a] { get => Query(a); set => Update(a, value); }
        public T this[int a, int b] => Query(a, b);
    }
}






//使わない
namespace Garbage
{
    /// <summary>
    /// 二分探索
    /// (基本はスニペットのものを使おう)
    /// </summary>
    static class BinarySearch
    {
        public static (long lb, long ub) Search(long lb, long ub, long item, Func<long, long, bool> comp) //compは midとアイテムを比較する
        {
            while (ub - lb > 1)
            {
                var mid = (lb + ub) >> 1;
                (lb, ub) = comp(mid, item) ? (mid, ub) : (lb, mid);
            }
            return (lb, ub);
        }
        public static (long lb, long ub) Search(long lb, long ub, long item) => Search(lb, ub, item, (mid, item) => mid < item); //下の境界を求める
        public static (int lb, int ub) Search(int lb, int ub, long item, Func<long, long, bool> comp) => Search(lb, ub, item, comp);
    }



    class Template2
    {
        //遅すぎ (Math.MaxMinの10倍遅い)
        public static T Max<T>(params T[] nums) where T : IComparable => nums.Aggregate((max, next) => max.CompareTo(next) < 0 ? next : max);
        public static T Min<T>(params T[] nums) where T : IComparable => nums.Aggregate((min, next) => min.CompareTo(next) > 0 ? next : min);
    }
}

