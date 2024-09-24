public class AssertMatchesAll extends RegexMatcher implements ValueAssertion<Matcher> { private final static String message = "argument '%s' value was: %s, did not match: %s, in tuple: %s"; @ConstructorProperties({"patternString"}) public AssertMatchesAll( String patternString ) { super( patternString, false ); } @ConstructorProperties({"patternString", "negateMatch"}) public AssertMatchesAll( String patternString, boolean negateMatch ) { super( patternString, negateMatch ); } @Override public boolean supportsPlannerLevel( PlannerLevel plannerLevel ) { return plannerLevel instanceof AssertionLevel; } @Override public void doAssert( FlowProcess flowProcess, ValueAssertionCall<Matcher> assertionCall ) { TupleEntry input = assertionCall.getArguments(); int pos = matchEachElementPos( assertionCall.getContext(), input ); if( pos != -1 ) BaseAssertion.throwFail( message, input.getFields().get( pos ), input.getObject( pos ), patternString, input.getTuple().print() ); } }