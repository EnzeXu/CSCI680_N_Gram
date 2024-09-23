public class CASValue<T> { private final long cas; private final T value; public CASValue(long c, T v) { super(); cas = c; value = v; } public long getCas() { return cas; } public T getValue() { return value; } @Override public String toString() { return "{CasValue " + cas + "/" + value + "}"; } }