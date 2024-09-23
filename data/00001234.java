public class Fields implements Comparable , Iterable < Comparable > , Serializable , Comparator < Tuple > { public static final Fields UNKNOWN = new Fields ( Kind . UNKNOWN ) ; public static final Fields NONE = new Fields ( Kind . NONE ) ; public static final Fields ALL = new Fields ( Kind . ALL ) ; public static final Fields GROUP = new Fields ( Kind . GROUP ) ; public static final Fields VALUES = new Fields ( Kind . VALUES ) ; public static final Fields ARGS = new Fields ( Kind . ARGS ) ; public static final Fields RESULTS = new Fields ( Kind . RESULTS ) ; public static final Fields REPLACE = new Fields ( Kind . REPLACE ) ; public static final Fields SWAP = new Fields ( Kind . SWAP ) ; public static final Fields FIRST = new Fields ( 0 ) ; public static final Fields LAST = new Fields ( -1 ) ; private static final int [ ] EMPTY_INT = new int [ 0 ] ; enum Kind { NONE , ALL , GROUP , VALUES , ARGS , RESULTS , UNKNOWN , REPLACE , SWAP } Comparable [ ] fields = new Comparable [ 0 ] ; boolean isOrdered = true ; Kind kind ; Type [ ] types ; Comparator [ ] comparators ; transient int [ ] thisPos ; transient Map < Comparable , Integer > index ; transient Map < Fields , int [ ] > posCache ; transient int hashCode ; public static Fields [ ] fields ( Fields . . . fields ) { return fields ; } public static Comparable [ ] names ( Comparable . . . names ) { return names ; } public static Type [ ] types ( Type . . . types ) { return types ; } public static Fields size ( int size ) { if ( size == 0 ) return Fields . NONE ; Fields fields = new Fields ( ) ; fields . kind = null ; fields . fields = expand ( size , 0 ) ; return fields ; } public static Fields size ( int size , Type type ) { if ( size == 0 ) return Fields . NONE ; Fields fields = new Fields ( ) ; fields . kind = null ; fields . fields = expand ( size , 0 ) ; for ( Comparable field : fields ) fields = fields . applyType ( field , type ) ; return fields ; } public static Fields join ( Fields . . . fields ) { return join ( false , fields ) ; } public static Fields join ( boolean maskDuplicateNames , Fields . . . fields ) { int size = 0 ; for ( Fields field : fields ) { if ( field . isSubstitution ( ) || field . isUnknown ( ) ) throw new TupleException ( "cannot join fields if one is a substitution or is unknown" ) ; size += field . size ( ) ; } if ( size == 0 ) return Fields . NONE ; Comparable [ ] elements = join ( size , fields ) ; if ( maskDuplicateNames ) { Set < String > names = new HashSet < String > ( ) ; for ( int i = elements . length - 1 ; i > = 0 ; i-- ) { Comparable element = elements [ i ] ; if ( names . contains ( element ) ) elements [ i ] = i ; else if ( element instanceof String ) names . add ( ( String ) element ) ; } } Type [ ] types = joinTypes ( size , fields ) ; if ( types == null ) return new Fields ( elements ) ; else return new Fields ( elements , types ) ; } private static Comparable [ ] join ( int size , Fields . . . fields ) { Comparable [ ] elements = expand ( size , 0 ) ; int pos = 0 ; for ( Fields field : fields ) { System . arraycopy ( field . fields , 0 , elements , pos , field . size ( ) ) ; pos += field . size ( ) ; } return elements ; } private static Type [ ] joinTypes ( int size , Fields . . . fields ) { Type [ ] elements = new Type [ size ] ; int pos = 0 ; for ( Fields field : fields ) { if ( field . isNone ( ) ) continue ; if ( field . types == null ) return null ; System . arraycopy ( field . types , 0 , elements , pos , field . size ( ) ) ; pos += field . size ( ) ; } return elements ; } public static Fields mask ( Fields fields , Fields mask ) { Comparable [ ] elements = expand ( fields . size ( ) , 0 ) ; System . arraycopy ( fields . fields , 0 , elements , 0 , elements . length ) ; for ( int i = elements . length - 1 ; i > = 0 ; i-- ) { Comparable element = elements [ i ] ; if ( element instanceof Integer ) continue ; if ( mask . getIndex ( ) . containsKey ( element ) ) elements [ i ] = i ; } return new Fields ( elements ) ; } public static Fields merge ( Fields . . . fields ) { List < Comparable > elements = new ArrayList < Comparable > ( ) ; List < Type > elementTypes = new ArrayList < Type > ( ) ; for ( Fields field : fields ) { Type [ ] types = field . getTypes ( ) ; int i = 0 ; for ( Comparable comparable : field ) { if ( !elements . contains ( comparable ) ) { elements . add ( comparable ) ; elementTypes . add ( types == null ? null : types [ i ] ) ; } i++ ; } } Comparable [ ] comparables = elements . toArray ( new Comparable [ elements . size ( ) ] ) ; Type [ ] types = elementTypes . toArray ( new Type [ elementTypes . size ( ) ] ) ; if ( Util . containsNull ( types ) ) return new Fields ( comparables ) ; return new Fields ( comparables , types ) ; } public static Fields copyComparators ( Fields toFields , Fields . . . fromFields ) { for ( Fields fromField : fromFields ) { for ( Comparable field : fromField ) { Comparator comparator = fromField . getComparator ( field ) ; if ( comparator != null ) toFields . setComparator ( field , comparator ) ; } } return toFields ; } public static Fields offsetSelector ( int size , int startPos ) { Fields fields = new Fields ( ) ; fields . kind = null ; fields . isOrdered = startPos == 0 ; fields . fields = expand ( size , startPos ) ; return fields ; } private static Comparable [ ] expand ( int size , int startPos ) { if ( size < 1 ) throw new TupleException ( "invalid size for fields : " + size ) ; if ( startPos < 0 ) throw new TupleException ( "invalid start position for fields : " + startPos ) ; Comparable [ ] fields = new Comparable [ size ] ; for ( int i = 0 ; i < fields . length ; i++ ) fields [ i ] = i + startPos ; return fields ; } public static Fields resolve ( Fields selector , Fields . . . fields ) { boolean hasUnknowns = false ; int size = 0 ; for ( Fields field : fields ) { if ( field . isUnknown ( ) ) hasUnknowns = true ; if ( !field . isDefined ( ) && field . isUnOrdered ( ) ) throw new TupleException ( "unable to select from field set : " + field . printVerbose ( ) ) ; size += field . size ( ) ; } if ( selector . isAll ( ) ) { Fields result = fields [ 0 ] ; for ( int i = 1 ; i < fields . length ; i++ ) result = result . append ( fields [ i ] ) ; return result ; } if ( selector . isReplace ( ) ) { if ( fields [ 1 ] . isUnknown ( ) ) throw new TupleException ( "cannot replace fields with unknown field declaration" ) ; if ( !fields [ 0 ] . contains ( fields [ 1 ] ) ) throw new TupleException ( "could not find all fields to be replaced , available : " + fields [ 0 ] . printVerbose ( ) + " , declared : " + fields [ 1 ] . printVerbose ( ) ) ; Type [ ] types = fields [ 0 ] . getTypes ( ) ; if ( types != null ) { for ( int i = 1 ; i < fields . length ; i++ ) { Type [ ] fieldTypes = fields [ i ] . getTypes ( ) ; if ( fieldTypes == null ) { fields [ 0 ] = fields [ 0 ] . applyTypes ( ( Type [ ] ) null ) ; } else { for ( int j = 0 ; j < fieldTypes . length ; j++ ) fields [ 0 ] = fields [ 0 ] . applyType ( fields [ i ] . get ( j ) , fieldTypes [ j ] ) ; } } } return fields [ 0 ] ; } if ( !selector . isDefined ( ) ) throw new TupleException ( "unable to use given selector : " + selector ) ; Set < String > notFound = new LinkedHashSet < String > ( ) ; Set < String > found = new HashSet < String > ( ) ; Fields result = size ( selector . size ( ) ) ; if ( hasUnknowns ) size = -1 ; Type [ ] types = null ; if ( size != -1 ) types = new Type [ result . size ( ) ] ; int offset = 0 ; for ( Fields current : fields ) { if ( current . isNone ( ) ) continue ; resolveInto ( notFound , found , selector , current , result , types , offset , size ) ; offset += current . size ( ) ; } if ( types != null && !Util . containsNull ( types ) ) result = result . applyTypes ( types ) ; notFound . removeAll ( found ) ; if ( !notFound . isEmpty ( ) ) throw new FieldsResolverException ( new Fields ( join ( size , fields ) ) , new Fields ( notFound . toArray ( new Comparable [ notFound . size ( ) ] ) ) ) ; if ( hasUnknowns ) return selector ; return result ; } private static void resolveInto ( Set < String > notFound , Set < String > found , Fields selector , Fields current , Fields result , Type [ ] types , int offset , int size ) { for ( int i = 0 ; i < selector . size ( ) ; i++ ) { Comparable field = selector . get ( i ) ; if ( field instanceof String ) { int index = current . indexOfSafe ( field ) ; if ( index == -1 ) notFound . add ( ( String ) field ) ; else result . set ( i , handleFound ( found , field ) ) ; if ( index != -1 && types != null && current . getType ( index ) != null ) types [ i ] = current . getType ( index ) ; continue ; } int pos = current . translatePos ( ( Integer ) field , size ) - offset ; if ( pos > = current . size ( ) || pos < 0 ) continue ; Comparable thisField = current . get ( pos ) ; if ( types != null && current . getType ( pos ) != null ) types [ i ] = current . getType ( pos ) ; if ( thisField instanceof String ) result . set ( i , handleFound ( found , thisField ) ) ; else result . set ( i , field ) ; } } private static Comparable handleFound ( Set < String > found , Comparable field ) { if ( found . contains ( field ) ) throw new TupleException ( "field name already exists : " + field ) ; found . add ( ( String ) field ) ; return field ; } public static Fields asDeclaration ( Fields fields ) { if ( fields == null ) return null ; if ( fields . isNone ( ) ) return fields ; if ( !fields . isDefined ( ) ) return UNKNOWN ; if ( fields . isOrdered ( ) ) return fields ; Fields result = size ( fields . size ( ) ) ; copy ( null , result , fields , 0 ) ; result . types = copyTypes ( fields . types , result . size ( ) ) ; result . comparators = fields . comparators ; return result ; } private static Fields asSelector ( Fields fields ) { if ( !fields . isDefined ( ) ) return UNKNOWN ; return fields ; } protected Fields ( Kind kind ) { this . kind = kind ; } public Fields ( ) { this . kind = Kind . NONE ; } @ ConstructorProperties ( { "fields" } ) public Fields ( Comparable . . . fields ) { if ( fields . length == 0 ) this . kind = Kind . NONE ; else this . fields = validate ( fields ) ; } @ ConstructorProperties ( { "field" , "type" } ) public Fields ( Comparable field , Type type ) { this ( names ( field ) , types ( type ) ) ; } @ ConstructorProperties ( { "fields" , "types" } ) public Fields ( Comparable [ ] fields , Type [ ] types ) { this ( fields ) ; if ( isDefined ( ) && types != null ) { if ( this . fields . length != types . length ) throw new IllegalArgumentException ( "given types array must be same length as fields" ) ; if ( Util . containsNull ( types ) ) throw new IllegalArgumentException ( "given types array contains null" ) ; this . types = copyTypes ( types , this . fields . length ) ; } } @ ConstructorProperties ( { "types" } ) public Fields ( Type . . . types ) { if ( types . length == 0 ) { this . kind = Kind . NONE ; return ; } this . fields = expand ( types . length , 0 ) ; if ( this . fields . length != types . length ) throw new IllegalArgumentException ( "given types array must be same length as fields" ) ; if ( Util . containsNull ( types ) ) throw new IllegalArgumentException ( "given types array contains null" ) ; this . types = copyTypes ( types , this . fields . length ) ; } public boolean isUnOrdered ( ) { return !isOrdered || kind == Kind . ALL ; } public boolean isOrdered ( ) { return isOrdered || kind == Kind . UNKNOWN ; } public boolean isDefined ( ) { return kind == null ; } public boolean isOutSelector ( ) { return isAll ( ) || isResults ( ) || isReplace ( ) || isSwap ( ) || isDefined ( ) ; } public boolean isArgSelector ( ) { return isAll ( ) || isNone ( ) || isGroup ( ) || isValues ( ) || isDefined ( ) ; } public boolean isDeclarator ( ) { return isUnknown ( ) || isNone ( ) || isAll ( ) || isArguments ( ) || isGroup ( ) || isValues ( ) || isDefined ( ) ; } public boolean isNone ( ) { return kind == Kind . NONE ; } public boolean isAll ( ) { return kind == Kind . ALL ; } public boolean isUnknown ( ) { return kind == Kind . UNKNOWN ; } public boolean isArguments ( ) { return kind == Kind . ARGS ; } public boolean isValues ( ) { return kind == Kind . VALUES ; } public boolean isResults ( ) { return kind == Kind . RESULTS ; } public boolean isReplace ( ) { return kind == Kind . REPLACE ; } public boolean isSwap ( ) { return kind == Kind . SWAP ; } public boolean isGroup ( ) { return kind == Kind . GROUP ; } public boolean isSubstitution ( ) { return isAll ( ) || isArguments ( ) || isGroup ( ) || isValues ( ) ; } private Comparable [ ] validate ( Comparable [ ] fields ) { isOrdered = true ; Set < Comparable > names = new HashSet < Comparable > ( ) ; for ( int i = 0 ; i < fields . length ; i++ ) { Comparable field = fields [ i ] ; if ( ! ( field instanceof String || field instanceof Integer ) ) throw new IllegalArgumentException ( String . format ( "invalid field type ( %s ) ; must be String or Integer : " , field ) ) ; if ( names . contains ( field ) ) throw new IllegalArgumentException ( "duplicate field name found : " + field ) ; names . add ( field ) ; if ( field instanceof Number && ( Integer ) field != i ) isOrdered = false ; } return fields ; } final Comparable [ ] get ( ) { return fields ; } public final Comparable get ( int i ) { return fields [ i ] ; } final void set ( int i , Comparable comparable ) { fields [ i ] = comparable ; if ( isOrdered ( ) && comparable instanceof Integer ) isOrdered = i == ( Integer ) comparable ; } public int [ ] getPos ( ) { if ( thisPos != null ) return thisPos ; if ( isAll ( ) || isUnknown ( ) ) thisPos = EMPTY_INT ; else thisPos = makeThisPos ( ) ; return thisPos ; } public boolean hasRelativePos ( ) { for ( int i : getPos ( ) ) { if ( i < 0 ) return true ; } return false ; } private int [ ] makeThisPos ( ) { int [ ] pos = new int [ size ( ) ] ; for ( int i = 0 ; i < size ( ) ; i++ ) { Comparable field = get ( i ) ; if ( field instanceof Number ) pos [ i ] = ( Integer ) field ; else pos [ i ] = i ; } return pos ; } private Map < Fields , int [ ] > getPosCache ( ) { if ( posCache == null ) posCache = new WeakHashMap < > ( ) ; return posCache ; } private final int [ ] putReturn ( Fields fields , int [ ] pos ) { getPosCache ( ) . put ( fields , pos ) ; return pos ; } public final int [ ] getPos ( Fields fields ) { return getPos ( fields , -1 ) ; } final int [ ] getPos ( Fields fields , int tupleSize ) { int [ ] pos = getPosCache ( ) . get ( fields ) ; if ( !isUnknown ( ) && pos != null ) return pos ; if ( fields . isAll ( ) ) return putReturn ( fields , null ) ; if ( isAll ( ) ) return putReturn ( fields , fields . getPos ( ) ) ; if ( size ( ) == 0 && isUnknown ( ) ) return translatePos ( fields , tupleSize ) ; pos = translatePos ( fields , size ( ) ) ; return putReturn ( fields , pos ) ; } private int [ ] translatePos ( Fields fields , int fieldSize ) { int [ ] pos = new int [ fields . size ( ) ] ; for ( int i = 0 ; i < fields . size ( ) ; i++ ) { Comparable field = fields . get ( i ) ; if ( field instanceof Number ) pos [ i ] = translatePos ( ( Integer ) field , fieldSize ) ; else pos [ i ] = indexOf ( field ) ; } return pos ; } final int translatePos ( Integer integer ) { return translatePos ( integer , size ( ) ) ; } final int translatePos ( Integer integer , int size ) { if ( size == -1 ) return integer ; if ( integer < 0 ) integer = size + integer ; if ( !isUnknown ( ) && ( integer > = size || integer < 0 ) ) throw new TupleException ( "position value is too large : " + integer + " , positions in field : " + size ) ; return integer ; } public int getPos ( Comparable fieldName ) { if ( fieldName instanceof Number ) return translatePos ( ( Integer ) fieldName ) ; else return indexOf ( fieldName ) ; } private final Map < Comparable , Integer > getIndex ( ) { if ( index != null ) return index ; Map < Comparable , Integer > local = new HashMap < Comparable , Integer > ( ) ; for ( int i = 0 ; i < size ( ) ; i++ ) local . put ( get ( i ) , i ) ; return index = local ; } private int indexOf ( Comparable fieldName ) { Integer result = getIndex ( ) . get ( fieldName ) ; if ( result == null ) throw new FieldsResolverException ( this , new Fields ( fieldName ) ) ; return result ; } int indexOfSafe ( Comparable fieldName ) { Integer result = getIndex ( ) . get ( fieldName ) ; if ( result == null ) return -1 ; return result ; } public Iterator iterator ( ) { return Arrays . stream ( fields ) . iterator ( ) ; } public Iterator < Fields > fieldsIterator ( ) { if ( types == null ) return Arrays . stream ( fields ) . map ( Fields : : new ) . iterator ( ) ; return IntStream . range ( 0 , fields . length ) . mapToObj ( pos - > new Fields ( fields [ pos ] , types [ pos ] ) ) . iterator ( ) ; } public Fields select ( Fields selector ) { if ( !isOrdered ( ) ) throw new TupleException ( "this fields instance can only be used as a selector" ) ; if ( selector . isAll ( ) ) return this ; if ( isUnknown ( ) ) return asSelector ( selector ) ; if ( selector . isNone ( ) ) return NONE ; Fields result = size ( selector . size ( ) ) ; for ( int i = 0 ; i < selector . size ( ) ; i++ ) { Comparable field = selector . get ( i ) ; if ( field instanceof String ) { result . set ( i , get ( indexOf ( field ) ) ) ; continue ; } int pos = translatePos ( ( Integer ) field ) ; if ( this . get ( pos ) instanceof String ) result . set ( i , this . get ( pos ) ) ; else result . set ( i , pos ) ; } if ( this . types != null ) { result . types = new Type [ result . size ( ) ] ; for ( int i = 0 ; i < selector . size ( ) ; i++ ) { Comparable field = selector . get ( i ) ; if ( field instanceof String ) result . setType ( i , getType ( indexOf ( field ) ) ) ; else result . setType ( i , getType ( translatePos ( ( Integer ) field ) ) ) ; } } return result ; } public Fields selectPos ( Fields selector ) { return selectPos ( selector , 0 ) ; } public Fields selectPos ( Fields selector , int offset ) { int [ ] pos = getPos ( selector ) ; Fields results = size ( pos . length ) ; for ( int i = 0 ; i < pos . length ; i++ ) results . fields [ i ] = pos [ i ] + offset ; return results ; } public Fields subtract ( Fields fields ) { if ( fields . isAll ( ) ) return Fields . NONE ; if ( fields . isNone ( ) ) return this ; List < Comparable > list = new LinkedList < Comparable > ( ) ; Collections . addAll ( list , this . get ( ) ) ; int [ ] pos = getPos ( fields , -1 ) ; for ( int i : pos ) list . set ( i , null ) ; Util . removeAllNulls ( list ) ; Type [ ] newTypes = null ; if ( this . types != null ) { List < Type > types = new LinkedList < Type > ( ) ; Collections . addAll ( types , this . types ) ; for ( int i : pos ) types . set ( i , null ) ; Util . removeAllNulls ( types ) ; newTypes = types . toArray ( new Type [ types . size ( ) ] ) ; } return new Fields ( list . toArray ( new Comparable [ list . size ( ) ] ) , newTypes ) ; } public Fields append ( Fields fields ) { return appendInternal ( fields , false ) ; } public Fields appendSelector ( Fields fields ) { return appendInternal ( fields , true ) ; } private Fields appendInternal ( Fields fields , boolean isSelect ) { if ( fields == null ) return this ; if ( this . isAll ( ) || fields . isAll ( ) ) throw new TupleException ( "cannot append fields : " + this . print ( ) + " + " + fields . print ( ) ) ; if ( ( this . isUnknown ( ) || this . size ( ) == 0 ) && fields . isUnknown ( ) ) return UNKNOWN ; if ( fields . isNone ( ) ) return this ; if ( this . isNone ( ) ) return fields ; Set < Comparable > names = new HashSet < Comparable > ( ) ; Fields result = size ( this . size ( ) + fields . size ( ) ) ; copyRetain ( names , result , this , 0 , isSelect ) ; copyRetain ( names , result , fields , this . size ( ) , isSelect ) ; if ( this . isUnknown ( ) || fields . isUnknown ( ) ) result . kind = Kind . UNKNOWN ; if ( ( this . isNone ( ) || this . types != null ) && fields . types != null ) { result . types = new Type [ this . size ( ) + fields . size ( ) ] ; if ( this . types != null ) System . arraycopy ( this . types , 0 , result . types , 0 , this . size ( ) ) ; System . arraycopy ( fields . types , 0 , result . types , this . size ( ) , fields . size ( ) ) ; } return result ; } public Fields rename ( Fields from , Fields to ) { if ( this . isSubstitution ( ) || this . isUnknown ( ) ) throw new TupleException ( "cannot rename fields in a substitution or unknown Fields instance : " + this . print ( ) ) ; if ( from . size ( ) != to . size ( ) ) throw new TupleException ( "from and to fields must be the same size" ) ; if ( from . isSubstitution ( ) || from . isUnknown ( ) ) throw new TupleException ( "from fields may not be a substitution or unknown" ) ; if ( to . isSubstitution ( ) || to . isUnknown ( ) ) throw new TupleException ( "to fields may not be a substitution or unknown" ) ; Comparable [ ] newFields = Arrays . copyOf ( this . fields , this . fields . length ) ; int [ ] pos = getPos ( from ) ; for ( int i = 0 ; i < pos . length ; i++ ) newFields [ pos [ i ] ] = to . fields [ i ] ; Type [ ] newTypes = null ; if ( this . types != null && to . types != null ) { newTypes = copyTypes ( this . types , this . size ( ) ) ; for ( int i = 0 ; i < pos . length ; i++ ) newTypes [ pos [ i ] ] = to . types [ i ] ; } return new Fields ( newFields , newTypes ) ; } public Fields rename ( BiFunction < Comparable , Type , Comparable > function ) { if ( this . isSubstitution ( ) || this . isUnknown ( ) ) throw new TupleException ( "cannot rename fields in a substitution or unknown Fields instance : " + this . print ( ) ) ; Comparable [ ] newFields = new Comparable [ this . fields . length ] ; for ( int i = 0 ; i < newFields . length ; i++ ) newFields [ i ] = function . apply ( this . fields [ i ] , this . types != null ? this . types [ i ] : null ) ; return new Fields ( newFields , this . types ) ; } public Fields rename ( Function < Comparable , Comparable > function ) { if ( this . isSubstitution ( ) || this . isUnknown ( ) ) throw new TupleException ( "cannot rename fields in a substitution or unknown Fields instance : " + this . print ( ) ) ; Comparable [ ] newFields = new Comparable [ this . fields . length ] ; for ( int i = 0 ; i < newFields . length ; i++ ) newFields [ i ] = function . apply ( this . fields [ i ] ) ; return new Fields ( newFields , this . types ) ; } public Fields renameString ( Function < String , Comparable > function ) { if ( this . isSubstitution ( ) || this . isUnknown ( ) ) throw new TupleException ( "cannot rename fields in a substitution or unknown Fields instance : " + this . print ( ) ) ; Comparable [ ] newFields = new Comparable [ this . fields . length ] ; for ( int i = 0 ; i < newFields . length ; i++ ) newFields [ i ] = function . apply ( this . fields [ i ] . toString ( ) ) ; return new Fields ( newFields , this . types ) ; } public Fields project ( Fields fields ) { if ( fields == null ) return this ; Fields results = size ( fields . size ( ) ) . applyTypes ( fields . getTypes ( ) ) ; for ( int i = 0 ; i < fields . fields . length ; i++ ) { if ( fields . fields [ i ] instanceof String ) results . fields [ i ] = fields . fields [ i ] ; else if ( this . fields [ i ] instanceof String ) results . fields [ i ] = this . fields [ i ] ; else results . fields [ i ] = i ; } return results ; } private static void copy ( Set < String > names , Fields result , Fields fields , int offset ) { for ( int i = 0 ; i < fields . size ( ) ; i++ ) { Comparable field = fields . get ( i ) ; if ( ! ( field instanceof String ) ) continue ; if ( names != null ) { if ( names . contains ( field ) ) throw new TupleException ( "field name already exists : " + field ) ; names . add ( ( String ) field ) ; } result . set ( i + offset , field ) ; } } private static void copyRetain ( Set < Comparable > names , Fields result , Fields fields , int offset , boolean isSelect ) { for ( int i = 0 ; i < fields . size ( ) ; i++ ) { Comparable field = fields . get ( i ) ; if ( !isSelect && field instanceof Integer ) continue ; if ( names != null ) { if ( names . contains ( field ) ) throw new TupleException ( "field name already exists : " + field ) ; names . add ( field ) ; } result . set ( i + offset , field ) ; } } public void verifyContains ( Fields fields ) { if ( isUnknown ( ) ) return ; try { getPos ( fields ) ; } catch ( TupleException exception ) { throw new TupleException ( "these fields " + print ( ) + " , do not contain " + fields . print ( ) ) ; } } public boolean contains ( Fields fields ) { try { getPos ( fields ) ; return true ; } catch ( Exception exception ) { return false ; } } public int compareTo ( Fields other ) { if ( other . size ( ) != size ( ) ) return other . size ( ) < size ( ) ? 1 : -1 ; for ( int i = 0 ; i < size ( ) ; i++ ) { int c = get ( i ) . compareTo ( other . get ( i ) ) ; if ( c != 0 ) return c ; } return 0 ; } public int compareTo ( Object other ) { if ( other instanceof Fields ) return compareTo ( ( Fields ) other ) ; else return -1 ; } public String print ( ) { return " [ " + toString ( ) + " ] " ; } public String printVerbose ( ) { String fieldsString = toString ( ) ; return " [ { " + ( isDefined ( ) ? size ( ) : "?" ) + " } : " + fieldsString + " ] " ; } @ Override public String toString ( ) { String string ; if ( isOrdered ( ) ) string = orderedToString ( ) ; else string = unorderedToString ( ) ; if ( types != null ) string += " | " + Util . join ( Util . simpleTypeNames ( types ) , " , " ) ; return string ; } private String orderedToString ( ) { StringBuffer buffer = new StringBuffer ( ) ; if ( size ( ) != 0 ) { int startIndex = get ( 0 ) instanceof Number ? ( Integer ) get ( 0 ) : 0 ; for ( int i = 0 ; i < size ( ) ; i++ ) { Comparable field = get ( i ) ; if ( field instanceof Number ) { if ( i + 1 == size ( ) || ! ( get ( i + 1 ) instanceof Number ) ) { if ( buffer . length ( ) != 0 ) buffer . append ( " , " ) ; if ( startIndex != i ) buffer . append ( startIndex ) . append ( " : " ) . append ( field ) ; else buffer . append ( i ) ; startIndex = i ; } continue ; } if ( i != 0 ) buffer . append ( " , " ) ; if ( field instanceof String ) buffer . append ( "\'" ) . append ( field ) . append ( "\'" ) ; else if ( field instanceof Fields ) buffer . append ( ( ( Fields ) field ) . print ( ) ) ; startIndex = i + 1 ; } } if ( kind != null ) { if ( buffer . length ( ) != 0 ) buffer . append ( " , " ) ; buffer . append ( kind ) ; } return buffer . toString ( ) ; } private String unorderedToString ( ) { StringBuffer buffer = new StringBuffer ( ) ; for ( Object field : get ( ) ) { if ( buffer . length ( ) != 0 ) buffer . append ( " , " ) ; if ( field instanceof String ) buffer . append ( "\'" ) . append ( field ) . append ( "\'" ) ; else if ( field instanceof Fields ) buffer . append ( ( ( Fields ) field ) . print ( ) ) ; else buffer . append ( field ) ; } if ( kind != null ) { if ( buffer . length ( ) != 0 ) buffer . append ( " , " ) ; buffer . append ( kind ) ; } return buffer . toString ( ) ; } public final int size ( ) { return fields . length ; } public Fields applyFields ( Comparable . . . fields ) { Fields result = new Fields ( fields ) ; if ( types == null ) return result ; if ( types . length != result . size ( ) ) throw new IllegalArgumentException ( "given number of field names must match current fields size" ) ; result . types = copyTypes ( types , types . length ) ; return result ; } public Fields applyType ( Comparable fieldName , Type type ) { if ( type == null ) throw new IllegalArgumentException ( "given type must not be null" ) ; int pos ; try { pos = getPos ( asFieldName ( fieldName ) ) ; } catch ( FieldsResolverException exception ) { throw new IllegalArgumentException ( "given field name was not found : " + fieldName , exception ) ; } Fields results = new Fields ( fields ) ; results . types = this . types == null ? new Type [ size ( ) ] : copyTypes ( this . types , this . types . length ) ; results . types [ pos ] = type ; return results ; } public Fields applyTypeToAll ( Type type ) { Fields result = new Fields ( fields ) ; if ( type == null ) return result ; Type [ ] copy = new Type [ result . size ( ) ] ; Arrays . fill ( copy , type ) ; result . types = copy ; return result ; } public Fields applyTypes ( Fields fields ) { Fields result = new Fields ( this . fields , this . types ) ; for ( Comparable field : fields ) result = result . applyType ( field , fields . getType ( fields . getPos ( field ) ) ) ; return result ; } public Fields applyTypes ( Type . . . types ) { Fields result = new Fields ( fields ) ; if ( types == null || types . length == 0 ) return result ; if ( types . length != size ( ) ) throw new IllegalArgumentException ( "given number of class instances must match fields size" ) ; for ( Type type : types ) { if ( type == null ) throw new IllegalArgumentException ( "type must not be null" ) ; } result . types = copyTypes ( types , types . length ) ; return result ; } public Fields unApplyTypes ( ) { return applyTypes ( ) ; } public Type getType ( Comparable fieldName ) { if ( !hasTypes ( ) ) return null ; return getType ( getPos ( fieldName ) ) ; } public Type getType ( int pos ) { if ( !hasTypes ( ) ) return null ; return this . types [ pos ] ; } public Class getTypeClass ( Comparable fieldName ) { if ( !hasTypes ( ) ) return null ; return getTypeClass ( getPos ( fieldName ) ) ; } public Class getTypeClass ( int pos ) { Type type = getType ( pos ) ; if ( type instanceof CoercibleType ) return ( ( CoercibleType ) type ) . getCanonicalType ( ) ; return ( Class ) type ; } protected void setType ( int pos , Type type ) { if ( type == null ) throw new IllegalArgumentException ( "type may not be null" ) ; this . types [ pos ] = type ; } public Type [ ] getTypes ( ) { return copyTypes ( types , size ( ) ) ; } public Class [ ] getTypesClasses ( ) { if ( types == null ) return null ; Class [ ] classes = new Class [ types . length ] ; for ( int i = 0 ; i < types . length ; i++ ) { if ( types [ i ] instanceof CoercibleType ) classes [ i ] = ( ( CoercibleType ) types [ i ] ) . getCanonicalType ( ) ; else classes [ i ] = ( Class ) types [ i ] ; } return classes ; } private static Type [ ] copyTypes ( Type [ ] types , int size ) { if ( types == null ) return null ; Type [ ] copy = new Type [ size ] ; if ( types . length != size ) throw new IllegalArgumentException ( "types array must be same size as fields array" ) ; System . arraycopy ( types , 0 , copy , 0 , size ) ; return copy ; } public final boolean hasTypes ( ) { return types != null ; } public void setComparator ( Comparable fieldName , Comparator comparator ) { if ( ! ( comparator instanceof Serializable ) ) throw new IllegalArgumentException ( "given comparator must be serializable" ) ; if ( comparators == null ) comparators = new Comparator [ size ( ) ] ; try { comparators [ getPos ( asFieldName ( fieldName ) ) ] = comparator ; } catch ( FieldsResolverException exception ) { throw new IllegalArgumentException ( "given field name was not found : " + fieldName , exception ) ; } } public void setComparators ( Comparator . . . comparators ) { if ( comparators . length != size ( ) ) throw new IllegalArgumentException ( "given number of comparator instances must match fields size" ) ; for ( Comparator comparator : comparators ) { if ( ! ( comparator instanceof Serializable ) ) throw new IllegalArgumentException ( "comparators must be serializable" ) ; } this . comparators = comparators ; } protected static Comparable asFieldName ( Comparable fieldName ) { if ( fieldName instanceof Fields ) { Fields fields = ( Fields ) fieldName ; if ( !fields . isDefined ( ) ) throw new TupleException ( "given Fields instance must explicitly declare one field name or position : " + fields . printVerbose ( ) ) ; fieldName = fields . get ( 0 ) ; } return fieldName ; } protected Comparator getComparator ( Comparable fieldName ) { if ( comparators == null ) return null ; try { return comparators [ getPos ( asFieldName ( fieldName ) ) ] ; } catch ( FieldsResolverException exception ) { return null ; } } public Comparator [ ] getComparators ( ) { Comparator [ ] copy = new Comparator [ size ( ) ] ; if ( comparators != null ) System . arraycopy ( comparators , 0 , copy , 0 , size ( ) ) ; return copy ; } public boolean hasComparators ( ) { return comparators != null ; } @ Override public int compare ( Tuple lhs , Tuple rhs ) { return lhs . compareTo ( comparators , rhs ) ; } @ Override public boolean equals ( Object object ) { if ( this == object ) return true ; if ( object == null || getClass ( ) != object . getClass ( ) ) return false ; Fields fields = ( Fields ) object ; return equalsFields ( fields ) && Arrays . equals ( types , fields . types ) ; } public boolean equalsFields ( Fields fields ) { return fields != null && this . kind == fields . kind && Arrays . equals ( get ( ) , fields . get ( ) ) ; } @ Override public int hashCode ( ) { if ( hashCode == 0 ) hashCode = Arrays . hashCode ( get ( ) ) ; return hashCode ; } }