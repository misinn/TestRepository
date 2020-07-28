using System;
using System.Collections.Generic;
using System.Linq;

class Graph<T>
{
    private List<T>[] G;
    public Graph(int size)
    {
        G = new List<T>[size];
        for (int i = 0; i < size; i++)
        {
            G[i] = new List<T>();
        }
    }

    public List<T> this[int i] => G[i];
    public void Add(int from, T to) => G[from].Add(to);
    public List<T>[] ToListArray() => G;
    public int Length => G.Length;
}

/*class WeightedGraph<T>
{
    
}
*/
class TopologicalSort
{   //ここからメソッドをコピペ
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

class Dijkstraa
{
    //ここからメソッドをコピペ
    static long[] Dijkstra(Graph<(int to, long cost)> G,int sp)
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
            foreach (var i in G[v])
            {
                if (d[i.to] > d[v] + i.cost)
                {
                    d[i.to] = d[v] + i.cost;
                    que.Push((i.to, d[i.to]));
                }
            }
        }
        return d;
    }

}

class 半分全列挙
{
    static long HalfFullEnumeration((long v, long w)[] Pairs, long W)
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
    private Graph<(int to, long cost)> MSTree;
    private List<(int u, int v, long cost)> es;
    private int V;
    public long costsum = 0;
    public MinimumSpanningTree(Graph<(int to, long cost)> G)
    {
        es = new List<(int u,int v, long cost)>();
        MSTree = new Graph<(int to, long cost)>(G.Length);
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
                MSTree.Add(e.u, (e.v, e.cost));
                MSTree.Add(e.v, (e.u, e.cost));
            }
        }
        return res;
    }

    public List<(int to,long cost)> this[int i] =>MSTree[i];
}

class Modular
{
    private const int M = 1000000007;
    private long value;
    public Modular(long value) { this.value = value; }
    public static implicit operator Modular(long a)
    {
        var m = a % M;
        return new Modular((m < 0) ? m + M : m);
    }
    public static Modular operator +(Modular a, Modular b)
    {
        return a.value + b.value;
    }
    public static Modular operator -(Modular a, Modular b)
    {
        return a.value - b.value;
    }
    public static Modular operator *(Modular a, Modular b)
    {
        return a.value * b.value;
    }
    private static Modular Pow(Modular a, int n)
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
    {
        return a * Pow(b, M - 2);
    }
    private static readonly List<int> facs = new List<int> { 1 };
    private static Modular Fac(int n)
    {
        for (int i = facs.Count; i <= n; ++i)
        {
            facs.Add((int)(Math.BigMul(facs.Last(), i) % M));
        }
        return facs[n];
    }
    public static Modular Ncr(int n, int r)
    {
        return (n < r) ? 0
             : (n == r) ? 1
                        : Fac(n) / (Fac(r) * Fac(n - r));
    }
    public static explicit operator int(Modular a)
    {
        return (int)a.value;
    }
}
