public class StringEdgeNameProvider<E> implements EdgeNameProvider<E> { public StringEdgeNameProvider() { } @Override public String getEdgeName( E edge ) { return edge.toString(); } }