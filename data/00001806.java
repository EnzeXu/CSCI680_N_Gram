public class RecordReaderIterator extends SingleValueCloseableIterator < RecordReader > { public RecordReaderIterator ( RecordReader reader ) { super ( reader ) ; } @ Override public void close ( ) throws IOException { getCloseableInput ( ) . close ( ) ; } }