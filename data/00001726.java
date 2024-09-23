public class TestLongComparator implements Hasher<Long>, StreamComparator<BufferedInputStream>, Comparator<Long>, Serializable { boolean reverse = true; public TestLongComparator() { } public TestLongComparator( boolean reverse ) { this.reverse = reverse; } @Override public int compare( Long lhs, Long rhs ) { if( lhs == null && rhs == null ) return 0; if( lhs == null ) return !reverse ? -1 : 1; if( rhs == null ) return !reverse ? 1 : -1; return reverse ? rhs.compareTo( lhs ) : lhs.compareTo( rhs ); } @Override public int compare( BufferedInputStream lhsStream, BufferedInputStream rhsStream ) { if( lhsStream == null && rhsStream == null ) return 0; if( lhsStream == null ) return !reverse ? -1 : 1; if( rhsStream == null ) return !reverse ? 1 : -1; HadoopTupleInputStream lhsInput = new HadoopTupleInputStream( lhsStream, new TupleSerialization().getElementReader() ); HadoopTupleInputStream rhsInput = new HadoopTupleInputStream( rhsStream, new TupleSerialization().getElementReader() ); try { Long l1 = (Long) lhsInput.readVLong(); Long l2 = (Long) rhsInput.readVLong(); return reverse ? l2.compareTo( l1 ) : l1.compareTo( l2 ); } catch( Exception exception ) { throw new CascadingException( exception ); } } @Override public int hashCode( Long value ) { if( value == null ) return 0; return value.hashCode(); } }