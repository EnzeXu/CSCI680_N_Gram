public class AssertNull extends BaseAssertion implements ValueAssertion { public AssertNull ( ) { super ( "argument '%s' value was not null , in tuple : %s" ) ; } @ Override public void doAssert ( FlowProcess flowProcess , ValueAssertionCall assertionCall ) { TupleEntry input = assertionCall . getArguments ( ) ; int pos = 0 ; for ( Object value : input . getTuple ( ) ) { if ( value != null ) fail ( input . getFields ( ) . get ( pos ) , input . getTuple ( ) . print ( ) ) ; pos++ ; } } }