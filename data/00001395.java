public class ForeverValueIterator<Value> extends SingleValueIterator<Value> { public ForeverValueIterator( Value value ) { super( value ); } @Override public Value next() { return value; } }