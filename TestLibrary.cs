using System;
using System.Collections.Generic;
using System.Linq;

class MyMath
{
    public static long GCD(long a, long b)
        => a == 0 ? b : GCD(b % a, a);
    public static long LCM(long a, long b)
        => a / GCD(a, b) * b;
    public static int ModPow(int a, int b, int mod) //未テスト　オーバーフローの可能性あり
    {
        if (b == 0) return 1;
        else if (b == 1) return a;
        else
        {
            int p = ModPow(a, b/2, mod);//p*p*ModPow(a,b%2) ,p=ModPow(a,b/2)
            return (int)(Math.BigMul((int)(Math.BigMul(p, p) % mod), ModPow(a, b % 2, mod))%mod);
        }

    }
    public static int Max(int a, int b, int c)
        => Math.Max(a, Math.Max(b, c));
    public static long Max(long a, long b, long c)
        => Math.Max(a, Math.Max(b, c));
    public static double Max(double a, double b, double c)
        => Math.Max(a, Math.Max(b, c));
    public static int Max(int a, int b, int c, int d)
        => Math.Max(a, Math.Max(b, Math.Max(c, d)));
    public static long Max(long a, long b, long c, long d)
        => Math.Max(a, Math.Max(b, Math.Max(c, d)));
    public static double Max(double a, double b, double c, double d)
        => Math.Max(a, Math.Max(b, Math.Max(c, d)));

    public static int Min(int a, int b, int c)
        => Math.Min(a, Math.Min(b, c));
    public static long Min(long a, long b, long c)
        => Math.Min(a, Math.Min(b, c));
    public static double Min(double a, double b, double c)
        => Math.Min(a, Math.Min(b, c));
    public static int Min(int a, int b, int c, int d)
        => Math.Max(a, Math.Min(b, Math.Min(c, d)));
    public static long Min(long a, long b, long c, long d)
        => Math.Max(a, Math.Min(b, Math.Min(c, d)));
    public static double Min(double a, double b, double c, double d)
        => Math.Max(a, Math.Min(b, Math.Min(c, d)));

}

class Scanner
{
    public static int RInt() => ReadStream<int>();
    public static long RLong() => ReadStream<long>();
    public static string RString() => Console.ReadLine();
    public static double RDouble() => ReadStream<double>();

    public static int[] RInts() => Array.ConvertAll(Console.ReadLine().Split(), int.Parse);
    public static int[] RInts(Func<string, int> func) => Console.ReadLine().Split().Select(func).ToArray();
    public static long[] RLongs() => Array.ConvertAll(Console.ReadLine().Split(), long.Parse);
    public static long[] RLongs(Func<string, long> func) => Console.ReadLine().Split().Select(func).ToArray();
    public static double[] RDoubles() => Array.ConvertAll(Console.ReadLine().Split(), double.Parse);
    public static double[] RDoubles(Func<string, double> func) => Console.ReadLine().Split().Select(func).ToArray();
    public static string[] RStrings() => Console.ReadLine().Split();
    //public static string[] RStrings(Func<string, string> func) => Console.ReadLine().Split().Select(func).ToArray();

    public static T1 ReadStream<T1>()
    {
        var r = RString();
        var r1 = (T1)Convert.ChangeType(r, typeof(T1));
        return r1;
    }
    public static (T1, T2) ReadStream<T1, T2>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        return (r1, r2);
    }
    public static (T1, T2, T3) ReadStream<T1, T2, T3>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        var r3 = (T3)Convert.ChangeType(r[2], typeof(T3));
        return (r1, r2, r3);
    }
    public static (T1, T2, T3, T4) ReadStream<T1, T2, T3, T4>()
    {
        var r = RStrings();
        var r1 = (T1)Convert.ChangeType(r[0], typeof(T1));
        var r2 = (T2)Convert.ChangeType(r[1], typeof(T2));
        var r3 = (T3)Convert.ChangeType(r[2], typeof(T3));
        var r4 = (T4)Convert.ChangeType(r[3], typeof(T4));
        return (r1, r2, r3, r4);
    }
}


class Graph<T>//有向グラフ
{
    protected List<T>[] G;
    public Graph(int size=200002)
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

class Graph<T,U> //重み付き有向グラフ
{
    private List<(T to, U cost)>[] G;
    public Graph(int size=200002)
    {
        G = new List<(T to, U cost)>[size];
        for (int i = 0; i < size; i++)
        {
            G[i] = new List<(T, U)>();
        }
    }

    public List<(T to,U cost)> this[int i] => G[i];
    public void AddEdge(int from, T to, U cost) => G[from].Add((to, cost));
    public List<(T to, U cost)>[] ToListArray() => G;
    public int Length => G.Length;
}

static class TopologicalSort
{   
    //Degreesはその頂点からのびる辺の数
    static List<int> Topologicalsort(Graph<int> G,int[] Degrees)
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
            foreach(var i in G[v])
            {
                Degrees[i]--;
                if (Degrees[i] == 0)
                    que.Enqueue(i);
            }
        }
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
    private Graph<int,long> MSTree;
    private List<(int u, int v, long cost)> es;
    private int V;
    public long costsum = 0;
    public MinimumSpanningTree(Graph<int,long> G)
    {
        es = new List<(int u,int v, long cost)>();
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

    public List<(int to,long cost)> this[int i] =>MSTree[i];
}

class Modular  //Modしながら計算する型
{
    private const int M = 1000000007;
    private long value;
    public Modular(long value = 0) { this.value = value; }
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
    public static Modular Pow(Modular a, int n)
    {
        switch (n)
        {
            case 0:
                return 1;
            case 1:
                return a;
            default:
                var p = Pow(a, n / 2);
                return p * p * Pow(a, n % 2);
        }
    }
    public static Modular operator /(Modular a, Modular b)
    => a * Pow(b, M - 2);
    
    private static readonly List<int> facs = new List<int> { 1 };
    public static Modular Fac(int n)   //階乗
    {
        for (int i = facs.Count; i <= n; ++i)
        {
            facs.Add((int)(Math.BigMul(facs.Last(), i) % M));
        }
        return facs[n];
    }
    public static Modular Fac(int r, int n)
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
                        : (n < 1000000) ? Fac(n) / (Fac(r) * Fac(n - r))
                            : Fac(n - r + 1, n) / Fac(r);
    }
    public static explicit operator int(Modular a)
    {
        return (int)a.value;
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
