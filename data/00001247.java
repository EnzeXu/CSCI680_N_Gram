public class TuplePair extends Tuple implements Resettable2 < Tuple , Tuple > { private final Tuple [ ] tuples = new Tuple [ 2 ] ; public static Tuple [ ] tuples ( TuplePair tuplePair ) { return tuplePair . tuples ; } public TuplePair ( ) { super ( ( List < Object > ) null ) ; tuples [ 0 ] = new Tuple ( ) ; tuples [ 1 ] = new Tuple ( ) ; } public TuplePair ( Tuple lhs , Tuple rhs ) { super ( ( List < Object > ) null ) ; tuples [ 0 ] = lhs ; tuples [ 1 ] = rhs ; if ( lhs == null ) throw new IllegalArgumentException ( "lhs may not be null" ) ; if ( rhs == null ) throw new IllegalArgumentException ( "rhs may not be null" ) ; } public Tuple getLhs ( ) { return tuples [ 0 ] ; } public Tuple getRhs ( ) { return tuples [ 1 ] ; } @ Override public void reset ( Tuple lhs , Tuple rhs ) { tuples [ 0 ] = lhs ; tuples [ 1 ] = rhs ; } @ Override public boolean equals ( Object object ) { if ( this == object ) return true ; if ( object == null || getClass ( ) != object . getClass ( ) ) return false ; TuplePair tuplePair = ( TuplePair ) object ; if ( !Arrays . equals ( tuples , tuplePair . tuples ) ) return false ; return true ; } @ Override public int hashCode ( ) { return Arrays . hashCode ( tuples ) ; } @ Override public int compareTo ( Object other ) { if ( other instanceof TuplePair ) return compareTo ( ( TuplePair ) other ) ; else return -1 ; } @ Override public int compareTo ( Tuple other ) { if ( other instanceof TuplePair ) return compareTo ( ( TuplePair ) other ) ; else return -1 ; } public int compareTo ( TuplePair tuplePair ) { int c = tuples [ 0 ] . compareTo ( tuplePair . tuples [ 0 ] ) ; if ( c != 0 ) return c ; c = tuples [ 1 ] . compareTo ( tuplePair . tuples [ 1 ] ) ; return c ; } @ Override public String toString ( ) { return tuples [ 0 ] . print ( ) + tuples [ 1 ] . print ( ) ; } @ Override public String print ( ) { return tuples [ 0 ] . print ( ) + tuples [ 1 ] . print ( ) ; } }