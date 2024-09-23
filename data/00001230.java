public class Tuple implements Comparable < Object > , Iterable < Object > , Serializable { public static final Tuple NULL = Tuples . asUnmodifiable ( new Tuple ( ) ) ; private final static String printDelim = "\t" ; protected transient boolean isUnmodifiable = false ; protected List < Object > elements ; protected Tuple ( List < Object > elements ) { this . elements = elements ; } public Tuple ( ) { this ( new ArrayList < > ( ) ) ; } @ ConstructorProperties ( { "tuple" } ) public Tuple ( Tuple tuple ) { this ( new ArrayList < > ( tuple . elements ) ) ; } @ ConstructorProperties ( { "values" } ) public Tuple ( Object . . . values ) { this ( new ArrayList < > ( values . length ) ) ; Collections . addAll ( elements , values ) ; } public static Tuple size ( int size ) { return size ( size , null ) ; } public static Tuple size ( int size , Comparable value ) { Tuple result = new Tuple ( new ArrayList < > ( size ) ) ; for ( int i = 0 ; i < size ; i++ ) result . elements . add ( value ) ; return result ; } public static List < Object > elements ( Tuple tuple ) { return tuple . elements ; } public boolean isUnmodifiable ( ) { return isUnmodifiable ; } public Object getObject ( int pos ) { return elements . get ( pos ) ; } public char getChar ( int pos ) { return Coercions . CHARACTER . coerce ( getObject ( pos ) ) ; } public String getString ( int pos ) { return Coercions . STRING . coerce ( getObject ( pos ) ) ; } public float getFloat ( int pos ) { return Coercions . FLOAT . coerce ( getObject ( pos ) ) ; } public double getDouble ( int pos ) { return Coercions . DOUBLE . coerce ( getObject ( pos ) ) ; } public int getInteger ( int pos ) { return Coercions . INTEGER . coerce ( getObject ( pos ) ) ; } public long getLong ( int pos ) { return Coercions . LONG . coerce ( getObject ( pos ) ) ; } public short getShort ( int pos ) { return Coercions . SHORT . coerce ( getObject ( pos ) ) ; } public boolean getBoolean ( int pos ) { return Coercions . BOOLEAN . coerce ( getObject ( pos ) ) ; } public Tuple get ( int [ ] pos ) { if ( pos == null || pos . length == 0 ) return new Tuple ( this ) ; Tuple results = new Tuple ( new ArrayList < > ( pos . length ) ) ; for ( int i : pos ) results . elements . add ( elements . get ( i ) ) ; return results ; } public Tuple get ( Fields declarator , Fields selector ) { try { return get ( getPos ( declarator , selector ) ) ; } catch ( Exception exception ) { throw new TupleException ( "unable to select from : " + declarator . print ( ) + " , using selector : " + selector . print ( ) , exception ) ; } } public int [ ] getPos ( Fields declarator , Fields selector ) { if ( !declarator . isUnknown ( ) && elements . size ( ) != declarator . size ( ) ) throw new TupleException ( "field declaration : " + declarator . print ( ) + " , does not match tuple : " + print ( ) ) ; return declarator . getPos ( selector , size ( ) ) ; } public Tuple leave ( int [ ] pos ) { verifyModifiable ( ) ; Tuple results = remove ( pos ) ; List < Object > temp = results . elements ; results . elements = this . elements ; this . elements = temp ; return results ; } public void clear ( ) { verifyModifiable ( ) ; elements . clear ( ) ; } public void add ( Comparable value ) { add ( ( Object ) value ) ; } public void add ( Object value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addBoolean ( boolean value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addShort ( short value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addInteger ( int value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addLong ( long value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addFloat ( float value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addDouble ( double value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addString ( String value ) { verifyModifiable ( ) ; elements . add ( value ) ; } public void addAll ( Object . . . values ) { verifyModifiable ( ) ; if ( values . length == 1 && values [ 0 ] instanceof Tuple ) addAll ( ( Tuple ) values [ 0 ] ) ; else Collections . addAll ( elements , values ) ; } public void addAll ( Tuple tuple ) { verifyModifiable ( ) ; if ( tuple != null ) elements . addAll ( tuple . elements ) ; } public void setAll ( Tuple tuple ) { verifyModifiable ( ) ; if ( tuple == null ) return ; for ( int i = 0 ; i < tuple . elements . size ( ) ; i++ ) internalSet ( i , tuple . elements . get ( i ) ) ; } public void setAll ( Tuple . . . tuples ) { verifyModifiable ( ) ; if ( tuples . length == 0 ) return ; int pos = 0 ; for ( int i = 0 ; i < tuples . length ; i++ ) { Tuple tuple = tuples [ i ] ; if ( tuple == null ) continue ; for ( int j = 0 ; j < tuple . elements . size ( ) ; j++ ) internalSet ( pos++ , tuple . elements . get ( j ) ) ; } } public void setAll ( Iterable < Tuple > tuples ) { verifyModifiable ( ) ; int pos = 0 ; for ( Tuple tuple : tuples ) { if ( tuple == null ) continue ; for ( int j = 0 ; j < tuple . elements . size ( ) ; j++ ) internalSet ( pos++ , tuple . elements . get ( j ) ) ; } } public void setAllTo ( Object value ) { verifyModifiable ( ) ; for ( int i = 0 ; i < elements . size ( ) ; i++ ) internalSet ( i , value ) ; } public void set ( int index , Object value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setBoolean ( int index , boolean value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setShort ( int index , short value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setInteger ( int index , int value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setLong ( int index , long value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setFloat ( int index , float value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setDouble ( int index , double value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } public void setString ( int index , String value ) { verifyModifiable ( ) ; internalSet ( index , value ) ; } protected final void internalSet ( int index , Object value ) { try { elements . set ( index , value ) ; } catch ( IndexOutOfBoundsException exception ) { if ( elements . size ( ) != 0 ) throw new TupleException ( "failed to set a value beyond the end of the tuple elements array , size : " + size ( ) + " , index : " + index ) ; else throw new TupleException ( "failed to set a value , tuple may not be initialized with values , is zero length" ) ; } } public void put ( Fields declarator , Fields fields , Tuple tuple ) { verifyModifiable ( ) ; int [ ] pos = getPos ( declarator , fields ) ; for ( int i = 0 ; i < pos . length ; i++ ) internalSet ( pos [ i ] , tuple . getObject ( i ) ) ; } public Tuple remove ( int [ ] pos ) { verifyModifiable ( ) ; int offset [ ] = new int [ pos . length ] ; for ( int i = 0 ; i < pos . length ; i++ ) { offset [ i ] = 0 ; for ( int j = 0 ; j < i ; j++ ) { if ( pos [ j ] < pos [ i ] ) offset [ i ] ++ ; } } Tuple results = new Tuple ( ) ; for ( int i = 0 ; i < pos . length ; i++ ) results . add ( elements . remove ( pos [ i ] - offset [ i ] ) ) ; return results ; } public Tuple remove ( Fields declarator , Fields selector ) { return remove ( getPos ( declarator , selector ) ) ; } Tuple extract ( int [ ] pos ) { Tuple results = new Tuple ( ) ; for ( int i : pos ) results . add ( elements . set ( i , null ) ) ; return results ; } Tuple nulledCopy ( int [ ] pos ) { if ( pos == null ) return size ( size ( ) ) ; Tuple results = new Tuple ( this ) ; for ( int i : pos ) results . set ( i , null ) ; return results ; } void set ( int [ ] pos , Tuple tuple ) { verifyModifiable ( ) ; if ( pos . length != tuple . size ( ) ) throw new TupleException ( "given tuple not same size as position array : " + pos . length + " , tuple : " + tuple . print ( ) ) ; int count = 0 ; for ( int i : pos ) elements . set ( i , tuple . elements . get ( count++ ) ) ; } private void set ( int [ ] pos , Type [ ] types , Tuple tuple , CoercibleType [ ] coercions ) { verifyModifiable ( ) ; if ( pos . length != tuple . size ( ) ) throw new TupleException ( "given tuple not same size as position array : " + pos . length + " , tuple : " + tuple . print ( ) ) ; int count = 0 ; for ( int i : pos ) { Object element = tuple . elements . get ( count ) ; if ( types != null ) { Type type = types [ i ] ; element = coercions [ count ] . coerce ( element , type ) ; } elements . set ( i , element ) ; count++ ; } } public void set ( Fields declarator , Fields selector , Tuple tuple ) { try { set ( declarator . getPos ( selector ) , declarator . getTypes ( ) , tuple , TupleEntry . getCoercions ( declarator , tuple ) ) ; } catch ( Exception exception ) { throw new TupleException ( "unable to set into : " + declarator . print ( ) + " , using selector : " + selector . print ( ) , exception ) ; } } protected void set ( Fields declarator , Fields selector , Tuple tuple , CoercibleType [ ] coercions ) { try { set ( declarator . getPos ( selector ) , declarator . getTypes ( ) , tuple , coercions ) ; } catch ( Exception exception ) { throw new TupleException ( "unable to set into : " + declarator . print ( ) + " , using selector : " + selector . print ( ) , exception ) ; } } public Iterator < Object > iterator ( ) { return elements . iterator ( ) ; } public boolean isEmpty ( ) { return elements . isEmpty ( ) ; } public int size ( ) { return elements . size ( ) ; } private Object [ ] elements ( ) { return elements . toArray ( ) ; } < T > T [ ] elements ( T [ ] destination ) { return elements . toArray ( destination ) ; } public Class [ ] getTypes ( ) { Class [ ] types = new Class [ elements . size ( ) ] ; for ( int i = 0 ; i < elements . size ( ) ; i++ ) { Object value = elements . get ( i ) ; if ( value != null ) types [ i ] = value . getClass ( ) ; } return types ; } public Tuple append ( Tuple . . . tuples ) { Tuple result = new Tuple ( this ) ; for ( Tuple tuple : tuples ) result . addAll ( tuple ) ; return result ; } public int compareTo ( Tuple other ) { if ( other == null || other . elements == null ) return 1 ; if ( other . elements . size ( ) != this . elements . size ( ) ) return this . elements . size ( ) - other . elements . size ( ) ; for ( int i = 0 ; i < this . elements . size ( ) ; i++ ) { Comparable lhs = ( Comparable ) this . elements . get ( i ) ; Comparable rhs = ( Comparable ) other . elements . get ( i ) ; if ( lhs == null && rhs == null ) continue ; if ( lhs == null ) return -1 ; else if ( rhs == null ) return 1 ; int c = lhs . compareTo ( rhs ) ; if ( c != 0 ) return c ; } return 0 ; } public int compareTo ( Comparator [ ] comparators , Tuple other ) { if ( comparators == null ) return compareTo ( other ) ; if ( other == null || other . elements == null ) return 1 ; if ( other . elements . size ( ) != this . elements . size ( ) ) return this . elements . size ( ) - other . elements . size ( ) ; if ( comparators . length != this . elements . size ( ) ) throw new IllegalArgumentException ( "comparator array not same size as tuple elements" ) ; for ( int i = 0 ; i < this . elements . size ( ) ; i++ ) { Object lhs = this . elements . get ( i ) ; Object rhs = other . elements . get ( i ) ; int c ; if ( comparators [ i ] != null ) c = comparators [ i ] . compare ( lhs , rhs ) ; else if ( lhs == null && rhs == null ) c = 0 ; else if ( lhs == null ) return -1 ; else if ( rhs == null ) return 1 ; else c = ( ( Comparable ) lhs ) . compareTo ( rhs ) ; if ( c != 0 ) return c ; } return 0 ; } public int compareTo ( Object other ) { if ( other instanceof Tuple ) return compareTo ( ( Tuple ) other ) ; else return -1 ; } @ SuppressWarnings ( { "ForLoopReplaceableByForEach" } ) @ Override public boolean equals ( Object object ) { if ( ! ( object instanceof Tuple ) ) return false ; Tuple other = ( Tuple ) object ; if ( this . elements . size ( ) != other . elements . size ( ) ) return false ; for ( int i = 0 ; i < this . elements . size ( ) ; i++ ) { Object lhs = this . elements . get ( i ) ; Object rhs = other . elements . get ( i ) ; if ( lhs == null && rhs == null ) continue ; if ( lhs == null || rhs == null ) return false ; if ( !lhs . equals ( rhs ) ) return false ; } return true ; } @ Override public int hashCode ( ) { int hash = 1 ; for ( Object element : elements ) hash = 31 * hash + ( element != null ? element . hashCode ( ) : 0 ) ; return hash ; } @ Override public String toString ( ) { return Util . join ( elements , printDelim , true ) ; } public String toString ( String delim ) { return Util . join ( elements , delim , true ) ; } public String toString ( String delim , boolean printNull ) { return Util . join ( elements , delim , printNull ) ; } public String format ( String format ) { return String . format ( format , elements ( ) ) ; } public String print ( ) { return printTo ( new StringBuffer ( ) ) . toString ( ) ; } public StringBuffer printTo ( StringBuffer buffer ) { buffer . append ( " [ " ) ; if ( elements != null ) { for ( int i = 0 ; i < elements . size ( ) ; i++ ) { Object element = elements . get ( i ) ; if ( element instanceof Tuple ) ( ( Tuple ) element ) . printTo ( buffer ) ; else if ( element == null ) buffer . append ( element ) ; else buffer . append ( "\'" ) . append ( element ) . append ( "\'" ) ; if ( i < elements . size ( ) - 1 ) buffer . append ( " , " ) ; } } buffer . append ( " ] " ) ; return buffer ; } private final void verifyModifiable ( ) { if ( isUnmodifiable ) throw new UnsupportedOperationException ( "this tuple is unmodifiable" ) ; } }