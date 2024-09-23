public class TupleEntry { private static final CoercibleType[] EMPTY_COERCIONS = new CoercibleType[ 0 ]; public static final TupleEntry NULL = new TupleEntry( Fields.NONE, Tuple.NULL ); private Fields fields; private Map<Class, CoercionFrom[]> iterableCache; private CoercibleType[] coercions = EMPTY_COERCIONS; private boolean isUnmodifiable = false; Tuple tuple; public static Tuple select( Fields selector, TupleEntry... entries ) { Tuple result = null; if( selector.isAll() ) { for( TupleEntry entry : entries ) { if( result == null ) result = entry.getTuple(); else result = result.append( entry.getTuple() ); } return result; } int size = 0; for( TupleEntry entry : entries ) size += entry.size(); result = Tuple.size( selector.size() ); int offset = 0; for( TupleEntry entry : entries ) { for( int i = 0; i < selector.size(); i++ ) { Comparable field = selector.get( i ); int pos; if( field instanceof String ) { pos = entry.fields.indexOfSafe( field ); if( pos == -1 ) continue; } else { pos = entry.fields.translatePos( (Integer) field, size ) - offset; if( pos >= entry.size() || pos < 0 ) continue; } result.set( i, entry.getObject( pos ) ); } offset += entry.size(); } return result; } public TupleEntry() { this.fields = Fields.NONE; setCoercions(); } @ConstructorProperties({"isUnmodifiable"}) public TupleEntry( boolean isUnmodifiable ) { this.fields = Fields.NONE; this.isUnmodifiable = isUnmodifiable; setCoercions(); } @ConstructorProperties({"fields"}) public TupleEntry( Fields fields ) { if( fields == null ) throw new IllegalArgumentException( "fields may not be null" ); this.fields = fields; setCoercions(); } @ConstructorProperties({"fields", "isUnmodifiable"}) public TupleEntry( Fields fields, boolean isUnmodifiable ) { if( fields == null ) throw new IllegalArgumentException( "fields may not be null" ); this.fields = fields; this.isUnmodifiable = isUnmodifiable; setCoercions(); } @ConstructorProperties({"fields", "tuple", "isUnmodifiable"}) public TupleEntry( Fields fields, Tuple tuple, boolean isUnmodifiable ) { if( fields == null ) throw new IllegalArgumentException( "fields may not be null" ); this.fields = fields; this.isUnmodifiable = isUnmodifiable; setTuple( tuple ); setCoercions(); } @ConstructorProperties({"fields", "tuple"}) public TupleEntry( Fields fields, Tuple tuple ) { if( fields == null ) throw new IllegalArgumentException( "fields may not be null" ); this.fields = fields; this.tuple = tuple; setCoercions(); } @ConstructorProperties({"tupleEntry"}) public TupleEntry( TupleEntry tupleEntry ) { if( tupleEntry == null ) throw new IllegalArgumentException( "tupleEntry may not be null" ); this.fields = tupleEntry.getFields(); this.tuple = tupleEntry.getTupleCopy(); setCoercions(); } @ConstructorProperties({"tuple"}) public TupleEntry( Tuple tuple ) { if( tuple == null ) throw new IllegalArgumentException( "tuple may not be null" ); this.fields = Fields.size( tuple.size() ); this.tuple = tuple; setCoercions(); } private void setCoercions() { if( coercions != EMPTY_COERCIONS ) return; coercions = getCoercions( getFields(), tuple ); } static CoercibleType[] getCoercions( Fields fields, Tuple tuple ) { Type[] types = fields.types; int size = fields.size(); size = size == 0 && tuple != null ? tuple.size() : size; if( size == 0 ) return EMPTY_COERCIONS; return Coercions.coercibleArray( size, types ); } public boolean isUnmodifiable() { return isUnmodifiable; } public Fields getFields() { return fields; } public boolean hasTypes() { return fields.hasTypes(); } public Tuple getTuple() { return tuple; } public Tuple getTupleCopy() { return new Tuple( tuple ); } public Tuple getCoercedTuple( Type[] types ) { return getCoercedTuple( types, Tuple.size( types.length ) ); } public Tuple getCoercedTuple( Type[] types, Tuple into ) { if( into == null ) throw new IllegalArgumentException( "into argument Tuple may not be null" ); if( coercions.length != types.length || types.length != into.size() ) throw new IllegalArgumentException( "current entry and given tuple and types must be same length" ); for( int i = 0; i < coercions.length; i++ ) { Object element = tuple.getObject( i ); into.set( i, Coercions.coerce( coercions[ i ], element, types[ i ] ) ); } return into; } public void setTuple( Tuple tuple ) { if( !isUnmodifiable && tuple != null && tuple.isUnmodifiable() ) throw new IllegalArgumentException( "current entry is modifiable but given tuple is not modifiable, make copy of given Tuple first" ); if( tuple != null && isUnmodifiable ) this.tuple = Tuples.asUnmodifiable( tuple ); else this.tuple = tuple; setCoercions(); } public void setCanonicalTuple( Tuple tuple ) { if( tuple == null ) { this.tuple = null; return; } if( isUnmodifiable ) tuple = Tuples.asUnmodifiable( tuple ); if( fields.size() != tuple.size() ) throw new IllegalArgumentException( "current entry and given tuple must be same length" ); for( int i = 0; i < coercions.length; i++ ) { Object element = tuple.getObject( i ); this.tuple.set( i, coercions[ i ].canonical( element ) ); } } public void setCanonicalValues( Object[] values ) { setCanonicalValues( values, 0, values.length ); } public void setCanonicalValues( Object[] values, int offset, int length ) { if( fields.size() != length ) throw new IllegalArgumentException( "current entry and given array must be same length" ); for( int i = offset; i < coercions.length; i++ ) { Object element = values[ i ]; this.tuple.set( i, coercions[ i ].canonical( element ) ); } } public int size() { return tuple.size(); } public Object getObject( int pos ) { return tuple.getObject( pos ); } public Object getObject( int pos, Type type ) { if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); return Coercions.coerce( coercions[ pos ], tuple.getObject( pos ), type ); } public Object getObject( Comparable fieldName ) { int pos = fields.getPos( asFieldName( fieldName ) ); return tuple.getObject( pos ); } public Object getObject( Comparable fieldName, Type type ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); return Coercions.coerce( coercions[ pos ], tuple.getObject( pos ), type ); } public void setRaw( int pos, Object value ) { tuple.set( pos, value ); } public void setRaw( Comparable fieldName, Object value ) { tuple.set( fields.getPos( asFieldName( fieldName ) ), value ); } public void setObject( Comparable fieldName, Object value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setBoolean( Comparable fieldName, boolean value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setShort( Comparable fieldName, short value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setInteger( Comparable fieldName, int value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setLong( Comparable fieldName, long value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setFloat( Comparable fieldName, float value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setDouble( Comparable fieldName, double value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public void setString( Comparable fieldName, String value ) { int pos = fields.getPos( asFieldName( fieldName ) ); if( pos > coercions.length - 1 ) throw new TupleException( "position value is too large: " + pos + ", positions in field: " + tuple.size() ); tuple.set( pos, coercions[ pos ].canonical( value ) ); } public String getString( Comparable fieldName ) { return (String) getObject( fieldName, String.class ); } public float getFloat( Comparable fieldName ) { return (Float) getObject( fieldName, float.class ); } public double getDouble( Comparable fieldName ) { return (Double) getObject( fieldName, double.class ); } public int getInteger( Comparable fieldName ) { return (Integer) getObject( fieldName, int.class ); } public long getLong( Comparable fieldName ) { return (Long) getObject( fieldName, long.class ); } public short getShort( Comparable fieldName ) { return (Short) getObject( fieldName, short.class ); } public boolean getBoolean( Comparable fieldName ) { return (Boolean) getObject( fieldName, boolean.class ); } private Comparable asFieldName( Comparable fieldName ) { return Fields.asFieldName( fieldName ); } public TupleEntry selectEntry( Fields selector ) { if( selector == null || selector.isAll() || fields == selector ) return this; if( selector.isNone() ) return isUnmodifiable ? TupleEntry.NULL : new TupleEntry(); return new TupleEntry( Fields.asDeclaration( selector ), tuple.get( this.fields, selector ), isUnmodifiable ); } public TupleEntry selectEntryCopy( Fields selector ) { if( selector == null || selector.isAll() || fields == selector ) return new TupleEntry( this ); if( selector.isNone() ) return new TupleEntry(); return new TupleEntry( Fields.asDeclaration( selector ), tuple.get( this.fields, selector ) ); } public Tuple selectTuple( Fields selector ) { if( selector == null || selector.isAll() || fields == selector ) return this.tuple; if( selector.isNone() ) return isUnmodifiable ? Tuple.NULL : new Tuple(); Tuple result = tuple.get( fields, selector ); if( isUnmodifiable ) Tuples.asUnmodifiable( result ); return result; } public Tuple selectTupleCopy( Fields selector ) { if( selector == null || selector.isAll() || fields == selector ) return new Tuple( this.tuple ); if( selector.isNone() ) return new Tuple(); return tuple.get( fields, selector ); } public Tuple selectInto( Fields selector, Tuple tuple ) { if( selector.isNone() ) return tuple; int[] pos = this.tuple.getPos( fields, selector ); if( pos == null || pos.length == 0 ) { tuple.addAll( this.tuple ); } else { for( int i : pos ) tuple.add( this.tuple.getObject( i ) ); } return tuple; } public void setTuple( Fields selector, Tuple tuple ) { if( selector == null || selector.isAll() ) this.tuple.setAll( tuple ); else this.tuple.set( fields, selector, tuple ); } public void set( TupleEntry tupleEntry ) { this.tuple.set( fields, tupleEntry.getFields(), tupleEntry.getTuple(), tupleEntry.coercions ); } public TupleEntry appendNew( TupleEntry entry ) { Fields appendedFields = fields.append( entry.fields.isUnknown() ? Fields.size( entry.tuple.size() ) : entry.fields ); Tuple appendedTuple = tuple.append( entry.tuple ); return new TupleEntry( appendedFields, appendedTuple ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof TupleEntry ) ) return false; TupleEntry that = (TupleEntry) object; if( fields != null ? !fields.equals( that.fields ) : that.fields != null ) return false; if( tuple != null ? fields.compare( tuple, that.tuple ) != 0 : that.tuple != null ) return false; return true; } @Override public int hashCode() { int result = fields != null ? fields.hashCode() : 0; result = 31 * result + ( tuple != null ? tuple.hashCode() : 0 ); return result; } @Override public String toString() { if( fields == null ) return "empty"; else if( tuple == null ) return "fields: " + fields.print(); else return "fields: " + fields.print() + " tuple: " + tuple.print(); } public <T> Iterable<T> asIterableOf( final Class<T> type ) { if( iterableCache == null ) iterableCache = new IdentityHashMap<>(); final CoercionFrom<Object, T>[] coerce = coercions.length == 0 ? null : iterableCache.computeIfAbsent( type, t -> Coercions.coercionsArray( type, coercions ) ); return () -> new Iterator<T>() { final Iterator<CoercionFrom<Object, T>> coercionsIterator = coerce == null ? new ForeverValueIterator<>( Coercions.OBJECT.to( type ) ) : Arrays.asList( coerce ).iterator(); final Iterator valuesIterator = tuple.iterator(); @Override public boolean hasNext() { return valuesIterator.hasNext(); } @Override public T next() { Object next = valuesIterator.next(); return coercionsIterator.next().coerce( next ); } @Override public void remove() { valuesIterator.remove(); } }; } public Iterable<String[]> asPairwiseIterable() { return () -> { final Iterator<Comparable> fieldsIterator = fields.iterator(); final Iterator<String> valuesIterator = asIterableOf( String.class ).iterator(); return new Iterator<String[]>() { @Override public boolean hasNext() { return valuesIterator.hasNext(); } @Override public String[] next() { String field = fieldsIterator.next().toString(); String next = valuesIterator.next(); return new String[]{field, next}; } @Override public void remove() { throw new UnsupportedOperationException( "remove is unsupported" ); } }; }; } }