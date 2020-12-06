using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MemoryMarshal = System.Runtime.InteropServices.MemoryMarshal;
class Scanner
{
    public static string RString() => Console.ReadLine();
    public static int RInt() => ReadTuple<int>();
    public static long RLong() => ReadTuple<long>();
    public static double RDouble() => ReadTuple<double>();
    public static string[] RStrings() => Console.ReadLine().Split();
    public static int[] RInts() => Array.ConvertAll(RStrings(), int.Parse);
    public static long[] RLongs() => Array.ConvertAll(RStrings(), long.Parse);
    public static double[] RDoubles() => Array.ConvertAll(RStrings(), double.Parse);
    public static int[] RInts(Func<int, int> func) => RInts().Select(func).ToArray();
    public static long[] RLongs(Func<long, long> func) => RLongs().Select(func).ToArray();
    public static double[] RDoubles(Func<double, double> func) => RDoubles().Select(func).ToArray();
    public static int[][] RIntss(int len) => new int[len][].Select(_ => RInts()).ToArray();
    public static long[][] RLongss(int len) => new long[len][].Select(_ => RLongs()).ToArray();
    public static double[][] RDoubless(int len) => new double[len][].Select(_ => RDoubles()).ToArray();
    public static int[][] RIntss(int len, Func<int, int> func) => new int[len][].Select(_ => RInts(func)).ToArray();
    public static long[][] RLongss(int len, Func<long, long> func) => new long[len][].Select(_ => RLongs(func)).ToArray();
    public static double[][] RDoubless(int len, Func<double, double> func) => new double[len][].Select(_ => RDoubles(func)).ToArray();
    public static T[,] R2DSplits<T>(int H, int W, Func<T, T> func = null, Action<int, int, string> action = null) //空白つなぎの入力
    {
        func ??= (_ => _);
        var ary = new T[H, W];
        for (int h = 0; h < H; h++)
        {
            var str = RStrings();
            for (int w = 0; w < W; w++)
            {
                ary[h, w] = func(ChangeType<T>(str[w]));
                action?.Invoke(h, w, str[w]);
            }
        }
        return ary;
    }
    public static T[,] R2DJoineds<T>(int H, int W, Func<char, T> func, Action<int, int, char> action = null) //文字列の入力
    {
        var ary = new T[H, W];
        for (int h = 0; h < H; h++)
        {
            var str = RString();
            for (int w = 0; w < W; w++)
            {
                ary[h, w] = func(str[w]);
                action?.Invoke(h, w, str[w]);
            }
        }
        return ary;
    }
    public static int[,] RInt2D(int H, int W, Func<int, int> func = null, Action<int, int, string> action = null) => R2DSplits(H, W, func, action);
    public static long[,] RLong2D(int H, int W, Func<long, long> func = null, Action<int, int, string> action = null) => R2DSplits(H, W, func, action);
    public static double[,] RDouble2D(int H, int W, Func<double, double> func = null, Action<int, int, string> action = null) => R2DSplits(H, W, func, action);
    public static bool[,] RBool2D(int H, int W, Func<char, bool> func, Action<int, int, char> action = null) => R2DJoineds(H, W, func, action);
    private static T ChangeType<T>(string r) => (T)Convert.ChangeType(r, typeof(T));
    public static T1 ReadTuple<T1>()
    {
        var r = RString();
        var r1 = (T1)Convert.ChangeType(r, typeof(T1));
        return r1;
    }
    public static (T1, T2) ReadTuple<T1, T2>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        return (r1, r2);
    }
    public static (T1, T2, T3) ReadTuple<T1, T2, T3>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        var r3 = (T3)Convert.ChangeType(r[2], typeof(T3));
        return (r1, r2, r3);
    }
    public static (T1, T2, T3, T4) ReadTuple<T1, T2, T3, T4>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        var r3 = (T3)Convert.ChangeType(r[2], typeof(T3));
        var r4 = (T4)Convert.ChangeType(r[3], typeof(T4));
        return (r1, r2, r3, r4);
    }
    public static (T1, T2, T3, T4, T5) ReadTuple<T1, T2, T3, T4, T5>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        var r3 = (T3)Convert.ChangeType(r[2], typeof(T3));
        var r4 = (T4)Convert.ChangeType(r[3], typeof(T4));
        var r5 = (T5)Convert.ChangeType(r[4], typeof(T5));
        return (r1, r2, r3, r4, r5);
    }
    public static T1[] ReadTuples<T1>(int N) //N行の入力 => T[N]出力
    {
        var res = new T1[N];
        for (int i = 0; i < N; i++)
        {
            res[i] = ChangeType<T1>(RString());
        }
        return res;
    }
    public static (T1[], T2[]) ReadTuples<T1, T2>(int N) //N行入力 => (T1[],T2[])
    {
        var (r1, r2) = (new T1[N], new T2[N]);
        for (int i = 0; i < N; i++)
        {
            (r1[i], r2[i]) = ReadTuple<T1, T2>();
        }
        return (r1, r2);
    }
    public static (T1[], T2[], T3[]) ReadTuples<T1, T2, T3>(int N)
    {
        var (r1, r2, r3) = (new T1[N], new T2[N], new T3[N]);
        for (int i = 0; i < N; i++)
        {
            (r1[i], r2[i], r3[i]) = ReadTuple<T1, T2, T3>();
        }
        return (r1, r2, r3);
    }
    public static (T1[], T2[], T3[], T4[]) ReadTuples<T1, T2, T3, T4>(int N)
    {
        var (r1, r2, r3, r4) = (new T1[N], new T2[N], new T3[N], new T4[N]);
        for (int i = 0; i < N; i++)
        {
            (r1[i], r2[i], r3[i], r4[i]) = ReadTuple<T1, T2, T3, T4>();
        }
        return (r1, r2, r3, r4);
    }
}
class Template
{
    public static long GCD(long a, long b)=> a == 0 ? b : GCD(b % a, a);
    public static long LCM(long a, long b)=> a / GCD(a, b) * b;
    public static bool ChMax<T>(ref T a, T b) where T :struct, IComparable<T> { if (a.CompareTo(b) > 0) { a = b; return true; } return false; }
    public static bool ChMin<T>(ref T a, T b) where T :struct, IComparable<T> { if (a.CompareTo(b) < 0) { a = b; return true; } return false; }
    public static T Max<T>(params T[] nums) where T : IComparable => nums.Aggregate((max, next) => max.CompareTo(next) < 0 ? next : max);
    public static T Min<T>(params T[] nums) where T : IComparable => nums.Aggregate((min, next) => min.CompareTo(next) > 0 ? next : min);
    public static void Copy<T>(T[] source, T[] destination) => Array.Copy(source, destination, source.Length);
    public static void Copy<T>(T[] source,T[] destination,int length)=>Array.Copy(source,destination,length);
    public static void Fill<T>(T[] ary, T init) => ary.AsSpan().Fill(init); 
    public static void Fill<T>(T[,] ary, T init) => MemoryMarshal.CreateSpan(ref ary[0, 0], ary.Length).Fill(init);
    public static void Fill<T>(T[,,] ary, T init) => MemoryMarshal.CreateSpan(ref ary[0, 0, 0], ary.Length).Fill(init);
    public static void Sort<T>(T[] ary) { Array.Sort(ary);}
    public static void Sort<T>(T[] ary, Comparison<T> comp) { Array.Sort(ary, comp); }
    public static void Sort<T>(T[] ary, IComparer<T> comp) { Array.Sort(ary, comp); }
    public static void Sort<T>(T[] sourse, params T[][] dest)
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
    public static void Reverse<T>(T[] ary) { Array.Reverse(ary); }
    public static string Join<T>(T[] ary, char separater = ' ') => string.Join(separater, ary);
    public static string Join<T>(T[] ary, string separater = " ") => string.Join(separater, ary);
    public static void WriteLine<T>(T str) { Console.WriteLine(str.ToString()); }
}

static class Debug //"using staticを置き換えることで入力をランダムにする" ことが目標
{
    public static Random Rand = new Random(DateTime.Now.Millisecond);
    public static int RandomInt(int min, int max) => Rand.Next(min, max);
    public static string RandomString(int min = 4, int max=8) => string.Join("", new char[RandomInt(min, max)].Select(_ => (char)('a' + RandomInt(0, 26))).ToArray());
    public static int[] RandomArray(int leng, int minvalue, int maxvalue)=>new int[leng].Select(_ => Rand.Next(minvalue, maxvalue)).ToArray();
    public static int RInt() => RandomInt(1, 10);
    public static int[] RInts() => RandomArray(10, 0, 10);
}

//文字列など 数の順列に変換する。
class IndexConverter<T> : IEnumerable<T> 
{
    Dictionary<T, int> itemToIndex = new Dictionary<T, int>();
    List<T> indexToItem = new List<T>();
    public int Add(T item)
    {
        if (itemToIndex.ContainsKey(item)) return -1;
        indexToItem.Add(item);
        return itemToIndex[item] = itemToIndex.Count;
    }
    public int IndexOf(T item) => itemToIndex[item];
    public IEnumerator<T> GetEnumerator() => indexToItem.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    public int this[T item] => itemToIndex[item];
    public T this[int index] => indexToItem[index];
    public int Count => itemToIndex.Count;
}

// Dictionaryの代わり 無いキーを参照してもエラーしない。
class Counter<TKey, TValue> : Dictionary<TKey, TValue> //Dictionary
{
    new public TValue this[TKey key]
    {
        set => base[key] = value;
        get => TryGetValue(key, out TValue value) ? value : default;
    }
}

// 二分探索 (基本はスニペットのものを使う)
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

// 有無向グラフ (Tは辺に持たせる情報) 
class Graph<T>
{
    protected List<Edge>[] G;
    public Graph(int size)
    {
        G = new List<Edge>[size].Select(_ => new List<Edge>()).ToArray();
    }
    public List<Edge> this[int i] => G[i];
    public void Add(int from, int to, T value = default) => G[from].Add(new Edge { From = from, To = to, Value = value });
    public void AddBoth(int u, int v, T value = default) { Add(u, v, value); Add(v, u, value); }
    public int Length => G.Length;
    public struct Edge
    {
        public int From, To;
        public T Value;
        public static implicit operator int(Edge edge) => edge.To;
    }
}

public class UnionFind : UnionFind<int>
{
    public UnionFind(int size) : base(size) { }
}

// 重み付きUF 親にマージした値がのる。(Tは頂点の情報)
public class UnionFind<T>
{
    int[] p;
    T[] data;
    Func<T, T, T> Merge;
    public UnionFind(int size, Func<T, T, T> merge, T[] init)
    {
        p = Enumerable.Repeat(-1, size).ToArray();
        data = init;
        Merge = merge;
    }
    public UnionFind(int size) : this(size, (a, b) => default, new T[size]) { }
    public bool Unite(int x, int y)
    {
        x = Root(x);
        y = Root(y);
        if (x != y)
        {
            if (p[y] < p[x]) (y, x) = (x, y);
            p[x] += p[y];
            p[y] = x;
            data[x] = Merge(data[x], data[y]);
        }
        return x != y;
    }
    public bool IsSameGroup(int x, int y) => Root(x) == Root(y);
    public int Root(int x) => p[x] < 0 ? x : p[x] = Root(p[x]);
    public int GetMem(int x) => -p[Root(x)];
    public T Weight(int x) => this[x];
    public T this[int x] => data[Root(x)]; //親の重み
}


// 重みつきUF 子に値を持たせる。(重みはlong型)
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

// 優先キュー
public class PriorityQueue<T> where T : IComparable
{
    public int Count { get; private set; } = 0;
    public int MaxSize { get; private set; } = 0;
    T[] m_heap;
    Comparison<T> Comp = null;
    public PriorityQueue(int maxSize, Comparison<T> comp)
    {
        if (maxSize <= 0) throw new Exception();
        MaxSize = maxSize;
        m_heap = new T[maxSize];
        Comp = comp;
    }
    public PriorityQueue() : this(16, (x, y) => x.CompareTo(y)) { }
    public PriorityQueue(int maxsize) : this(maxsize, (x, y) => x.CompareTo(y)) { }
    public PriorityQueue(Comparison<T> comp) : this(16, comp) { }
    public void Push(T x)
    {
        if (Count == MaxSize)
        {
            T[] new_heap = new T[MaxSize <<= 1];
            Array.Copy(m_heap, new_heap, m_heap.Length);
            m_heap = new_heap;
        }
        long i = Count++;
        while (i > 0)
        {
            long p = (i - 1) / 2;
            if (Comp(m_heap[p], x) <= 0) break;
            m_heap[i] = m_heap[p];
            i = p;
        }
        m_heap[i] = x;
    }
    public T Pop()
    {
        if (Count == 0) throw new Exception("Queue is empty.");
        T result = m_heap[0];
        T x = m_heap[--Count];
        long i = 0;
        while (i * 2 + 1 < Count)
        {
            long c1 = i * 2 + 1, c2 = i * 2 + 2;
            if (c2 < Count && Comp(m_heap[c2], m_heap[c1]) < 0) c1 = c2;
            if (Comp(m_heap[c1], x) >= 0) break;
            m_heap[i] = m_heap[c1];
            i = c1;
        }
        m_heap[i] = x;
        return result;
    }
}

// ダイクストラ グラフ上でspからそれぞれの点の最小コスト
class Dijkstraa 
{
    public static long[] Search(Graph<long> G, int sp,Comparison<(int to,long cost)> comp) //costで比較すればいい
    {
        var d = Enumerable.Repeat(long.MaxValue/4, G.Length).ToArray();
        var que = new PriorityQueue<(int to, long cost)>(500000, comp);
        d[sp] = 0;
        que.Push((sp, 0));
        while (que.Count > 0)
        {
            var p = que.Pop();
            int v = p.to;
            if (d[v] < p.cost) continue;
            foreach (var edge in G[v])
            {
                int to = edge.To;
                long cost = edge.Value;
                if (d[to] > d[v] + cost)
                {
                    que.Push((to, d[to] = d[v] + cost));
                }
            }
        }
        return d;
    }
    public static long[] Search(Graph<long> G, int sp) => Search(G, sp, (x, y) => x.cost.CompareTo(y.cost));
    public static long Search(Graph<long> G, int sp, int gp)=> Search(G, sp)[gp];
    public static long Search(Graph<long> G, int sp, int gp, Comparison<(int to,long cost)> comp) => Search(G, sp, comp)[gp];
}

// 最小全域木 経路のコストの和など(無向グラフのみ)
class MinimumSpanningTree 
{
    Graph<long> MSTree;
    List<Graph<long>.Edge> edges;
    int V;
    public long CostSum { get; private set; } = 0;
    public MinimumSpanningTree(Graph<long> G,Comparison<Graph<long>.Edge> comp = null)
    {
        comp ??= (a, b) => a.Value.CompareTo(b.Value);
        edges = new List<Graph<long>.Edge>();
        MSTree = new Graph<long>(G.Length);
        for (int i = 0; i < G.Length; i++)
        {
            foreach (Graph<long>.Edge edge in G[i])
            {
                edges.Add(edge);
            }
        }
        V = G.Length;
        CostSum = Kruskal(comp);
    }
    private long Kruskal(Comparison<Graph<long>.Edge> comp)
    {
        edges.Sort(comp);
        var union = new UnionFind(V);
        long res = 0;
        for (int i = 0; i < edges.Count; i++)
        {
            var e = edges[i];
            if (union.IsSameGroup(e.From, e.To)) continue;
            union.Unite(e.From, e.To);
            MSTree.AddBoth(e.From, e.To, e.Value);
            res += e.Value;
        }
        return res;
    }
    public List<Graph<long>.Edge> this[int i] => MSTree[i];
    public Graph<long> Graph => MSTree;
}

// 1000000007でModした計算
struct Modular
{
    const int M = 1000000007;
    const int arysize = 2000001;
    long value;
    public Modular(long value = 0) { this.value = value; }
    public override string ToString(){ return value.ToString(); }
    public static implicit operator Modular(long a)
    {
        var m = a % M;
        return new Modular((m < 0) ? m + M : m);
    }
    public static Modular operator +(Modular a, Modular b)=> a.value + b.value;
    public static Modular operator -(Modular a, Modular b)=> a.value - b.value;
    public static Modular operator *(Modular a, Modular b)=> a.value * b.value;
    public static Modular operator /(Modular a, Modular b)=> a * Pow(b, M - 2);
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
        return Fac(n) / Fac(n - r);
    }
    public static explicit operator int(Modular a)
    {
        return (int)a.value;
    }
}

// 行列(正方行列のみ)
class Mat 
{
    long[,] mat;
    static readonly long Mod = 1000000007;
    public Mat(int _size)
    {
        Size = _size;
        mat = new long[Size, Size];
    }
    public Mat(int[,] _mat)
    {
        Size = _mat.Length;
        mat = new long[Size, Size];
        Array.Copy(_mat, mat, Size * Size);
    }
    public int Size { get; }
    public long this[int i, int j]
    {
        get => mat[i, j];
        set { mat[i, j] = value;mat[i, j] %= Mod; }
    }
    public static Mat operator +(Mat a, Mat b)
    {
        if (a.Size != b.Size) throw new Exception($"ex at'+' a.size={a.Size} b.size={b.Size}");
        for (int i = 0; i < a.Size; i++)
        {
            for (int j = 0; j < a.Size; j++)
            {
                a[i, j] = (a[i, j] + b[i, j]) % Mod;
            }
        }
        return a;
    }
    public static Mat operator -(Mat a, Mat b)
    {
        if (a.Size != b.Size) throw new Exception($"ex at'-' a.size={a.Size} b.size={b.Size}");
        for (int i = 0; i < a.Size; i++)
        {
            for (int j = 0; j < a.Size; j++)
            {
                a[i, j] = (a[i, j] - b[i, j] + Mod) % Mod;
            }
        }
        return a;
    }
    public static Mat operator *(Mat a, Mat b)
    {
        if (a.Size != b.Size) throw new Exception($"ex at'*' a.size={a.Size} b.size={b.Size}");
        var C = new Mat(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            for (int k = 0; k < b.Size; k++)
            {
                for (int j = 0; j < a.Size; j++)
                {
                    C[i, j] = (C[i, j] + a[i, k] * b[k, j]) % Mod;
                }
            }
        }
        return C;
    }
    public static Mat operator *(Mat a,long b)
    {
        var C = new Mat(a.Size);
        for (int i = 0; i < a.Size; i++)
        {
            for (int j = 0; j < a.Size; j++)
            {
                C[i, j] = a[i, j] * b % Mod;
            }
        }
        return C;
    }
    public static Mat Pow(Mat A, long n)
    {
        Mat B = new Mat(A.Size);
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

// 半分全列挙 個数40のナップザック問題
static class HalfFullEnumeration
{
    //重さW以下で最大の価値
    public static long Knapsack(long[] v, long[] w, long W) //TODOpairsの無駄な要素を省く
    {
        int N = v.Length;
        int n1 = N / 2;
        int n2 = N - n1;
        var pairs1 = new (long v, long w)[n1];
        for (int i = 0; i < (1 << n1); i++)
        {
            long sv = 0, sw = 0;
            for (int j = 0; j < n1; j++)
            {
                if ((i & (1 << j)) == 0) continue;
                sv += v[i];
                sw += w[i];
            }
            pairs1[i] = (sv, sw);
        }
        Array.Sort(pairs1);
        var pairs2 = new (long v, long w)[n2];
        for (int i = 0; i < (1 << n2); i++)
        {
            long sv = 0, sw = 0;
            for (int j = n1; j < N; j++)
            {
                if ((i & (1 << j)) == 0) continue;
                sv += v[i];
                sw += w[i];
            }
            pairs2[i] = (sv, sw);
        }
        Array.Sort(pairs2);
        long res = 0;
        for (int i = 0; i < (1 << n2); i++)
        {
            var (lb, ub) = (-1, pairs2.Length);
            while (ub - lb > 1)
            {
                int mid = (ub + lb) / 2;
                if (pairs2[i].w + pairs1[mid].w <= W) lb = mid;
                else ub = mid;
            }
            if (lb == -1) continue;
            res = Math.Max(res, pairs2[i].v + pairs1[lb].v);
        }
        return res;
    }
}

// トポソ
static class TopologicalSort
{
    //Degreesはその頂点にのびる辺の数 無理ならnullを返す
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
                Degrees[i]--;
                if (Degrees[i] == 0) que.Enqueue(i);
            }
        }
        if (Degrees.Length != sorted.Count) return null;
        return sorted;
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

// BIT 区間の和 速めのLog(N)
class BIT
{
    int[] bit;
    int N;
    public BIT(int n)
    {
        bit = new int[n + 1];
        N = n;
    }
    public int Sum(int i)//[0,i)
    {
        int s = 0;
        while (i > 0)
        {
            s += bit[i];
            i -= (i & -i);
        }
        return s;
    }
    public void Add(int i, int x)
    {
        i++;
        while (i <= N)
        {
            bit[i] += x;
            i += (i & (-i));
        }
    }
    public int this[int i]
    {
        set => Add(i, value - this[i]);
        get => Sum(i + 1) - Sum(i);
    }
}

// 二次元BIT 二次元の区間の和
class BIT2D
{
    public int X, Y;
    int[,] bit;
    public BIT2D(int X,int Y)
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
    public void Add(int x,int y,int value)
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

// 単更新・範囲検索
/// <summary>
/// updfunc: Update時に上の区間に記録する下の区間同士の計算
/// qfunc: Query時に出力するための下の区間同士の比較
/// 上の二つは基本同じ式入れてOK...?
/// </summary>
class SegTree<T>
{
    int N;
    T init;
    T[] dat;
    Func<T, T, T> updFunc, qFunc;
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
    public T this[int a] { get => Query(a);set => Update(a, value); }
    public T this[int a,int b]=> Query(a,b);
}

// 遅延評価セグメントツリー  区間加算のみ対応(ModもOK)
class LazySegTree
{
    int n;
    long[] Data, Lazy;
    Func<long, long, long> func;
    public LazySegTree(int[] v, Func<long, long, long> _func)
    {
        int size = v.Length;
        func = _func;
        n = 1; while (n < size) n <<= 1;
        Data = new long[n * 2 - 1];
        Lazy = new long[n * 2 - 1];
        for (int i = 0; i < size; i++)
        {
            Data[i + size - 1] = v[i];
        }
        for (int i = n - 2; i >= 0; i--)//二段目n-1から(一段目は2n-1)
        {
            Data[i] = Data[i * 2 + 1] + Data[i * 2 + 2];
        }
    }
    public LazySegTree(int size, Func<long, long, long> _func) : this(new int[size], _func) { }
    void eval(int k, int l, int r)//nodeが呼び出された時に伝達する。
    {
        if (Lazy[k] != 0)
        {
            Data[k] += Lazy[k];
            if (r - l > 1)
            {
                Lazy[k * 2 + 1] += Lazy[k] >> 1;
                Lazy[k * 2 + 2] += Lazy[k] >> 1;
            }
        }
        Lazy[k] = 0;
    }
    public void Update(int a, long x) => Update(a, a + 1, x, 0, 0, n);
    public void Update(int a, int b, long x) => Update(a, b, x, 0, 0, n); //[a,b)
    private void Update(int a, int b, long x, int k, int l, int r)
    {
        eval(k, l, r);
        if (b <= l || r <= a) return;
        if (a <= l && r <= b)//完全にl,rが含まれる
        {
            Lazy[k] += (r - l) * x;
            eval(k, l, r);
        }
        else//どっちか片方範囲外
        {
            Update(a, b, x, k * 2 + 1, l, (l + r) >> 1);
            Update(a, b, x, k * 2 + 2, (l + r) >> 1, r);
            Data[k] = Data[k * 2 + 1] + Data[k * 2 + 2];
        }
    }
    public long Query(int a, int b) => Query(a, b, 0, 0, n);
    public long Query(int a) => Query(a, a + 1, 0, 0, n);
    private long Query(int a, int b, int k, int l, int r)
    {
        if (b <= l || r <= a) return 0;
        eval(k, l, r);
        if (a <= l && r <= b) return Data[k];
        else
        {
            var vl = Query(a, b, k * 2 + 1, l, (l + r) >> 1);
            var vr = Query(a, b, k * 2 + 2, (l + r) >> 1, r);
            return func(vl, vr);
        }
    }
    public long this[int a] { get => Query(a);set => Update(a, value); }
    public long this[int a,int b] { get => Query(a, b); set => Update(a, b, value); }
}


// 両端キュー 双方向からpushできるqueue
class Deque<T>
{
    T[] array;
    int offset, size;
    public int Count { get; private set; }
    public Deque(int size)
    {
        array = new T[this.size = size];
        Count = 0; offset = 0;
    }
    public Deque() : this(16) { }
    public T this[int index] { get => array[GetIndex(index)]; set => array[GetIndex(index)] = value; }
    int GetIndex(int index)
    {
        var tmp = index + offset;
        return tmp >= size ? tmp - size : tmp;
    }
    public T PeekFront() => array[offset];
    public T PeekBack() => array[GetIndex(Count - 1)];
    public void PushFront(T item)
    {
        if (Count == size) Extend();
        if (--offset < 0) offset += array.Length;
        array[offset] = item;
        Count++;
    }
    public void PushBack(T item)
    {
        if (Count == size) Extend();
        var id = (Count++) + offset;
        if (id >= size) id -= size;
        array[id] = item;
    }
    public T PopFront()
    {
        Count--;
        var tmp = array[offset++];
        if (offset >= size) offset -= size;
        return tmp;
    }
    public T PopBack() => array[GetIndex(--Count)];
    public void Insert(int index, T item)
    {
        PushFront(item);
        for (var i = 0; i < index; i++) this[i] = this[i + 1];
        this[index] = item;
    }
    public T RemoveAt(int index)
    {
        var tmp = this[index];
        for (var i = index; i > 0; i--) this[i] = this[i - 1];
        PopFront();
        return tmp;
    }
    void Extend()
    {
        var newArray = new T[size << 1];
        if (offset > size - Count)
        {
            var length = array.Length - offset;
            Array.Copy(array, offset, newArray, 0, length);
            Array.Copy(array, 0, newArray, length, Count - length);
        }
        else Array.Copy(array, offset, newArray, 0, Count);
        array = newArray;
        offset = 0;
        size <<= 1;
    }
}

// 累積和 [a,b)の計算 (get時に計算する。)
public class CumulativeSum
{
    long[] D, A;
    bool IsCalcd = false;
    Func<long, long, long> func;
    public CumulativeSum(int[] A, Func<long, long, long> func)
    {
        D = new long[A.Length + 1];
        Array.Copy(A, this.A = new long[A.Length], A.Length);
        this.func = func;
    }
    public CumulativeSum(int[] A) : this(A, (i, j) => i + j) { }
    public CumulativeSum(int size, Func<long, long, long> func) : this(new int[size], func) { }
    public CumulativeSum(int size) : this(new int[size]) { }
    private void Calc()
    {
        if (IsCalcd) return;
        for (int i = 0; i < A.Length; i++) D[i + 1] = func(D[i], A[i]);
        IsCalcd = true;
    }
    public long this[int i]{get { Calc(); return D[i + 1] - D[i]; } set { A[i] = (int)value; IsCalcd = false; }}
    public long this[int from, int to]{get { Calc(); return D[to] - D[from]; }}
    public int Length => D.Length - 1;
}

/// <summary>
/// ビットDPのサンプル
/// bitをテーブルにのせてDPするだけ(メモ化再帰)
/// n:地点数    d:距離行列
/// S:訪れた地点 v:現在地点
/// </summary>

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





