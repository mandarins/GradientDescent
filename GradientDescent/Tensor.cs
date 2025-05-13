using System;

public class Tensor
{
    public double[] Data { get; }
    public int Length => Data.Length;

    public Tensor(int size)
    {
        Data = new double[size];
    }

    public Tensor(double[] data)
    {
        Data = (double[])data.Clone();
    }

    public double this[int i]
    {
        get => Data[i];
        set => Data[i] = value;
    }

    // Element-wise addition
    public static Tensor operator +(Tensor a, Tensor b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Tensors must be the same length.");
        var result = new Tensor(a.Length);
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] + b[i];
        return result;
    }

    // Element-wise subtraction
    public static Tensor operator -(Tensor a, Tensor b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Tensors must be the same length.");
        var result = new Tensor(a.Length);
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] - b[i];
        return result;
    }

    // Scalar multiplication
    public static Tensor operator *(Tensor a, double scalar)
    {
        var result = new Tensor(a.Length);
        for (int i = 0; i < a.Length; i++)
            result[i] = a[i] * scalar;
        return result;
    }

    public static Tensor operator *(double scalar, Tensor a) => a * scalar;

    // Dot product
    public double Dot(Tensor other)
    {
        if (Length != other.Length)
            throw new ArgumentException("Tensors must be the same length.");
        double sum = 0;
        for (int i = 0; i < Length; i++)
            sum += this[i] * other[i];
        return sum;
    }

    // Clone
    public Tensor Clone() => new Tensor(Data);

    // ToString override for easy display
    public override string ToString() => $"[{string.Join(", ", Data)}]";
}
