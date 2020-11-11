using System;
using System.Collections.Generic;
using System.Linq;

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
    //public static string[] RStrings(Func<string, string> func) => Console.ReadLine().Split().Select(func).ToArray();
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
}


static class Template
{
    public static long GCD(long a, long b)
        => a == 0 ? b : GCD(b % a, a);
    public static long LCM(long a, long b)
        => a / GCD(a, b) * b;
    public static bool ChMax<T>(ref T a, T b) where T : IComparable<T> { if (a.CompareTo(b) > 0) { a = b; return true; } return false; }
    public static bool ChMin<T>(ref T a, T b) where T : IComparable<T> { if (a.CompareTo(b) < 0) { a = b; return true; } return false; }
    public static T Max<T>(params T[] nums) where T : IComparable => nums.Aggregate((max, next) => max.CompareTo(next) < 0 ? next : max);
    public static T Min<T>(params T[] nums) where T : IComparable => nums.Aggregate((min, next) => min.CompareTo(next) > 0 ? next : min);
    public static T[] Sort<T>(T[] ary) { Array.Sort(ary);return ary; }
    public static T[] Sort<T>(T[] ary, Comparison<T> comp) { Array.Sort(ary, comp); return ary; }
    public static T[] Sort<T>(T[] ary, IComparer<T> comp) { Array.Sort(ary, comp); return ary; }
    public static T[] Reverse<T>(T[] ary) { Array.Reverse(ary); return ary; }
}
static class Debug
{
    public static Random Rand = new Random(DateTime.Now.Millisecond);
    public static int RandomInt(int min, int max) => Rand.Next(min, max);
    public static string RandomString(int min = 4, int max=8) => string.Join("", new char[RandomInt(min, max)].Select(_ => (char)('a' + RandomInt(0, 26))).ToArray());
    public static int[] RandomArray(int leng, int minvalue, int maxvalue)=>new int[leng].Select(_ => Rand.Next(minvalue, maxvalue)).ToArray();
    public static int RInt() => RandomInt(1, 10);
    public static int[] RInts() => RandomArray(10, 0, 10);
}

class IndexConverter<T> //文字列など 数の順列に変換する。
{
    Dictionary<T, int> itemToIndex = new Dictionary<T, int>();
    List<T> indexToItem = new List<T>();

    public T GetItem(int i) => indexToItem[i];
    public int GetIndex(T item) => itemToIndex[item];
    public void Add(T item)
    {
        if (itemToIndex.ContainsKey(item)) return;
        itemToIndex[item] = itemToIndex.Count;
        indexToItem.Add(item);
    }
}

class Counter<T> //valueをlongに限定したDictionary キーがない値を参照してもエラーしない 初期化 0
{
    private Dictionary<T, long> dic;
    public Counter()
    {
        dic = new Dictionary<T, long>();
    }
    public long this[T item]
    {
        set
        {
            dic.TryGetValue(item, out long v);
            dic[item] = value;
        }
        get
        {
            dic.TryGetValue(item, out long v);
            return v;
        }
    }
    public IEnumerator<KeyValuePair<T, long>> GetEnumerator()
    {
        foreach (var item in dic)
        {
            yield return item;
        }
    }
    public bool ContainsKey(T item) => dic.ContainsKey(item);
    public Dictionary<T, long>.KeyCollection Keys => dic.Keys;
    public Dictionary<T, long>.ValueCollection Values => dic.Values;
}


class Graph<T>//有向グラフ
{
    protected List<T>[] G;
    public Graph(int size = 200002)
    {
        
        G = new List<T>[size];
        for (int i = 0; i < size; i++)
        {
            G[i] = new List<T>();
        }
    }

    public List<T> this[int i] => G[i];
    public void AddEdge(int from, T to) => G[from].Add(to);
    public List<T>[] ToListArray() => G;
    public int Length => G.Length;
}



class Graph<T, U> //重み付き有向グラフ
{
    List<(T to, U cost)>[] G;
    public Graph(int size = 200002)
    {
        G = new List<(T to, U cost)>[size];
        for (int i = 0; i < size; i++)
        {
            G[i] = new List<(T, U)>();
        }
    }

    public List<(T to, U cost)> this[int i] => G[i];
    public void AddEdge(int from, T to, U cost) => G[from].Add((to, cost));
    public List<(T to, U cost)>[] ToListArray() => G;
    public int Length => G.Length;
}


static class TopologicalSort
{
    //Degreesはその頂点にのびる辺の数 無理ならnullを返す
    static List<int> Topologicalsort(Graph<int> G, int[] Degrees)
    {
        var que = new Queue<int>();
        for (int i = 0; i < Degrees.Length; i++)
        {
            if (Degrees[i] == 0)
                que.Enqueue(i);
        }

        var sorted = new List<int>();
        while (que.Count > 0)
        {
            var v = que.Dequeue();
            sorted.Add(v);
            foreach (var i in G[v])
            {
                Degrees[i]--;
                if (Degrees[i] == 0)
                    que.Enqueue(i);
            }
        }
        if (Degrees.Length != sorted.Count) return null;
        return sorted;
    }
}


public class Union_Find
{
    private int[] data;
    public Union_Find(int size)
    {
        data = new int[size];
        for (int i = 0; i < size; i++) data[i] = -1;
    }
    public bool Unite(int x, int y)
    {
        x = Root(x);
        y = Root(y);
        if (x != y)
        {
            if (data[y] < data[x])
            {
                var tmp = y;
                y = x;
                x = tmp;
            }
            data[x] += data[y];
            data[y] = x;
        }
        return x != y;
    }
    public bool IsSameGroup(int x, int y) => Root(x) == Root(y);
    public int Root(int x) => data[x] < 0 ? x : data[x] = Root(data[x]);
    public int getMem(int x) => -data[Root(x)];
}


public class PriorityQueue<T>
{
    public long Size { get; private set; } = 0;
    public long MaxSize { get; private set; } = 0;
    public T[] m_heap;
    private Comparison<T> Comp = null;

    public PriorityQueue(long maxSize, Comparison<T> comp)
    {
        if (maxSize <= 0)
            throw new Exception();
        MaxSize = maxSize;
        m_heap = new T[maxSize];

        Comp = comp;
    }

    public void Push(T x)
    {
        if (Size == MaxSize)
        {
            T[] new_heap = new T[MaxSize << 1];
            Array.Copy(m_heap, new_heap, MaxSize);
            m_heap = new_heap;
            MaxSize <<= 1;
        }

        long i = Size++;
        while (i > 0)
        {
            long p = (i - 1) / 2;
            if (Comp(m_heap[p], x) <= 0)
                break;
            m_heap[i] = m_heap[p];
            i = p;
        }
        m_heap[i] = x;
    }

    public T Pop()
    {
        if (Size == 0)
            throw new Exception("Queue is empty.");

        T result = m_heap[0];
        T x = m_heap[--Size];
        long i = 0;
        while (i * 2 + 1 < Size)
        {
            long c1 = i * 2 + 1, c2 = i * 2 + 2;
            if (c2 < Size && Comp(m_heap[c2], m_heap[c1]) < 0)
                c1 = c2;
            if (Comp(m_heap[c1], x) >= 0)
                break;
            m_heap[i] = m_heap[c1];
            i = c1;
        }
        m_heap[i] = x;

        return result;
    }
}


static class Dijkstraa
{
    public static long[] Search(Graph<int, long> G, int sp)
    {
        //sp からスタート
        var d = Enumerable.Repeat(long.MaxValue, G.Length).ToArray();
        d[sp] = 0;
        var que = new PriorityQueue<(int to, long cost)>(500000, (x, y) => x.cost.CompareTo(y.cost));
        que.Push((sp, 0));
        while (que.Size > 0)
        {
            var p = que.Pop();
            int v = p.to;
            if (d[v] < p.cost) continue;
            foreach (var (to, cost) in G[v])
            {
                if (d[to] > d[v] + cost)
                {
                    d[to] = d[v] + cost;
                    que.Push((to, d[to]));
                }
            }
        }
        return d;
    }
    public static long Search(Graph<int, long> G, int sp, int gp)
        => Search(G, sp)[gp];
}


static class 半分全列挙
{
    public static long HalfFullEnumeration((long v, long w)[] Pairs, long W)
    {
        int N = Pairs.Length;
        int n2 = N / 2;
        var ps = new List<(long v, long w)>(1 << n2);

        for (int i = 0; i < (1 << n2); i++)
        {
            long sv = 0, sw = 0;
            for (int j = 0; j < n2; j++)
            {
                if ((i & (1 << j)) > 0)
                {
                    sv += Pairs[j].v;
                    sw += Pairs[j].w;
                }
            }
            ps.Add((sv, sw));
        }

        ps.Sort((x, y) => x.w.CompareTo(y.w));

        int m = 1;
        for (int i = 1; i < (1 << n2); i++)
        {
            if (ps[m - 1].v < ps[i].v)
                ps[m++] = ps[i];
        }
        ps.RemoveRange(m, ps.Count - m);

        long res = 0; ;
        for (int i = 0; i < (1 << (N - n2)); i++)
        {
            long sv = 0, sw = 0;
            for (int j = 0; j < (N - n2); j++)
            {
                if ((i & (1 << j)) > 0)
                {
                    sv += Pairs[n2 + j].v;
                    sw += Pairs[n2 + j].w;
                }
            }

            int ub = ps.Count;
            int lb = -1;
            while (ub - lb > 1)
            {
                int mid = (ub + lb) / 2;
                if (sw + ps[mid].w > W) ub = mid;
                else lb = mid;
            }
            if (lb == -1) continue;
            else res = Math.Max(res, sv + ps[lb].v);
        }

        return res;
    }
}


class MinimumSpanningTree
{
    private Graph<int, long> MSTree;
    private List<(int u, int v, long cost)> es;
    private int V;
    public long costsum = 0;
    public MinimumSpanningTree(Graph<int, long> G)
    {
        es = new List<(int u, int v, long cost)>();
        MSTree = new Graph<int, long>(G.Length);
        for (int i = 0; i < G.Length; i++)
        {
            foreach (var j in G[i])
            {
                es.Add((i, j.to, j.cost));
            }
        }
        V = G.Length;
        costsum = kruskal();
    }

    private long kruskal()
    {
        es.Sort((x, y) => x.cost.CompareTo(y.cost));
        var union = new Union_Find(V);
        long res = 0;
        for (int i = 0; i < es.Count; i++)
        {
            var e = es[i];
            if (!union.IsSameGroup(e.u, e.v))
            {
                union.Unite(e.u, e.v);
                res += e.cost;
                MSTree.AddEdge(e.u, e.v, e.cost);
                MSTree.AddEdge(e.v, e.u, e.cost);
            }
        }
        return res;
    }

    public List<(int to, long cost)> this[int i] => MSTree[i];
}


class Modular
{
    private const int M = 1000000007;
    private const int arysize = 2000001;
    private long value;
    public Modular(long value = 0) { this.value = value; }
    public override string ToString(){ return value.ToString(); }
    public static implicit operator Modular(long a)
    {
        var m = a % M;
        return new Modular((m < 0) ? m + M : m);
    }
    public static Modular operator +(Modular a, Modular b)
    => a.value + b.value;
    public static Modular operator -(Modular a, Modular b)
    => a.value - b.value;
    public static Modular operator *(Modular a, Modular b)
    => a.value * b.value;
    public static Modular Pow(Modular a, long n)
    {
        Modular ans = 1;
        for (; n > 0; n >>= 1, a *= a)
        {
            if ((n & 1) == 1) ans *= a;
        }
        return ans;
    }
    public static Modular operator /(Modular a, Modular b)
    {
        return a * Pow(b, M - 2);
    }
    private static int[] facs = new int[arysize];
    private static int facscount = -1;

    public static Modular Fac(int n)   //階乗
    {
        facs[0] = 1;
        while (++facscount <= n)
        {
            facs[facscount + 1] = (int)(Math.BigMul(facs[facscount], facscount + 1) % M);
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
        r = n - r;
        return Fac(n) / Fac(r);
    }
    public static explicit operator int(Modular a)
    {
        return (int)a.value;
    }
}

class Mat
{
    private readonly long[][] mat;
    private static readonly long Mod = 1000000007;
    public Mat(int _size)
    {
        Size = _size;
        mat = new long[Size][];
        for (int i = 0; i < Size; i++)
        {
            mat[i] = new long[Size];
        }
    }
    public int Size { get; }
    public long this[int i, int j]
    {
        set
        {
            mat[i][j] = value;
        }
        get
        {
            return mat[i][j];
        }
    }
    public static Mat operator +(Mat a, Mat b)
    {
        if (a.Size != b.Size) throw new Exception($"行列のサイズが違います。");
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
        if (a.Size != b.Size) throw new Exception($"行列のサイズが違います。");
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
        if (a.Size != b.Size) throw new Exception($"行列のサイズが違います。");
        Mat C = new Mat(a.Size);
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

class ZAlgorithm//先頭文字列と何文字一致しているか
{
    private static string S;
    private int[] Same;
    public ZAlgorithm(string s)
    {
        S = s;
        Same = Search();
    }
    static int[] Search()
    {
        int N = S.Length;
        int c = 0;
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


class BIT//区間の和をlogNで求める
{
    private int[] bit;
    private int N;
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


//練習用RMQ
//これを修正して他のクエリに対応させる
class SegTree
{
    static readonly int MAX_N = 1 << 17;
    static int n;
    static int[] dat = new int[2 * MAX_N - 1];

    public SegTree(int N)
    {
        n = 1;
        while (n <= N) n *= 2;
        for (int i = 0; i < 2 * n - 1; i++)//初期値
        {
            dat[i] = int.MaxValue;
        }
    }
    //節ごとにその範囲の答えをいれる。
    public void Update(int k, int a)
    {
        k += n - 1;
        dat[k] = a;
        while (k > 0)
        {
            k = (k - 1) / 2;
            dat[k] = Math.Min(dat[k * 2 + 1], dat[k * 2 + 2]);
        }
    }
    //[a,b)の最小値を求めるクエリ
    //[l,r)区間と配列のk番目が対応している。
    public int Query(int a, int b) => Query(a, b, 0, 0, n);
    private int Query(int a, int b, int k, int l, int r)
    {
        if (r <= a || b <= l) return int.MaxValue;//範囲が重ならないとき
        else if (a <= l && r <= b) return dat[k];//範囲が完全に含まれるとき
        else//どっちかが範囲外の時
        {
            int vl = Query(a, b, k * 2 + 1, l, (l + r) / 2);
            int vr = Query(a, b, k * 2 + 2, (l + r) / 2, r);
            return Math.Min(vl, vr);
        }
    }
}

//遅延評価セグ木
//区間加算と区間和の計算 どちらもO(logN)
class LazySegment
{
    int n;
    long[] Data, Lazy;

    public LazySegment(long[] v)
    {
        int N = v.Length;
        n = 1; while (n < N) n *= 2;
        Data = new long[n * 2 - 1];
        Lazy = new long[n * 2 - 1];
        for (int i = 0; i < N; i++)
        {
            Data[i + n - 1] = v[i];
        }
        for (int i = n - 2; i >= 0; i--)//二段目n-1から(一段目は2n-1)
        {
            Data[i] = Data[i * 2 + 1] + Data[i * 2 + 2];
        }
    }
    public LazySegment(int size):this(new long[size])
    {
    }
    void eval(int k, int l, int r)//nodeが呼び出された時に伝達する。
    {
        if (Lazy[k] != 0)
        {
            Data[k] += Lazy[k];
            if (r - l > 1)
            {
                Lazy[k * 2 + 1] += Lazy[k] / 2;
                Lazy[k * 2 + 2] += Lazy[k] / 2;
            }
        }
        Lazy[k] = 0;
    }
    public void Update(int a, int b, int x) => Update(a, b, x, 0, 0, n);
    public void Update(int a, int b, int x, int k, int l, int r)
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
            Update(a, b, x, k * 2 + 1, l, (l + r) / 2);
            Update(a, b, x, k * 2 + 2, (l + r) / 2, r);
            Data[k] = Data[k * 2 + 1] + Data[k * 2 + 1];
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
            var vl = Query(a, b, k * 2 + 1, l, (l + r) / 2);
            var vr = Query(a, b, k * 2 + 2, (l + r) / 2, r);
            return vl + vr;
        }
    }
}
