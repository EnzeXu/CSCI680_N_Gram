public class SingleCloseableInputIterator extends SingleValueCloseableIterator<Closeable> { public SingleCloseableInputIterator( Closeable input ) { super( input ); } @Override public void close() throws IOException { getCloseableInput().close(); } }