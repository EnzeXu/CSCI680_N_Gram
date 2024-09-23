public class AssertNotEquals extends BaseAssertion implements ValueAssertion { private Tuple values; @ConstructorProperties({"values"}) public AssertNotEquals( Object... values ) { super( values == null ? 1 : values.length, "argument tuple: %s was not equal to values: %s" ); if( values == null ) throw new IllegalArgumentException( "values may not be null" ); if( values.length == 0 ) throw new IllegalArgumentException( "values may not be empty" ); this.values = new Tuple( values ); } @Property(name = "values", visibility = Visibility.PRIVATE) @PropertyDescription("The expected values.") public Collection getValues() { return Tuples.asCollection( values ); } @Override public void doAssert( FlowProcess flowProcess, ValueAssertionCall assertionCall ) { Tuple tuple = assertionCall.getArguments().getTuple(); if( tuple.equals( values ) ) fail( tuple.print(), values.print() ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof AssertNotEquals ) ) return false; if( !super.equals( object ) ) return false; AssertNotEquals that = (AssertNotEquals) object; if( values != null ? !values.equals( that.values ) : that.values != null ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( values != null ? values.hashCode() : 0 ); return result; } }