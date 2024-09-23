public class AssertMatches extends RegexMatcher implements ValueAssertion<Matcher> { private final static String message = "argument tuple: %s did not match: %s"; @ConstructorProperties({"patternString"}) public AssertMatches( String patternString ) { super( patternString, false ); } @ConstructorProperties({"patternString", "delimiter"}) public AssertMatches( String patternString, String delimiter ) { super( patternString, false, delimiter ); } @ConstructorProperties({"patternString", "negateMatch"}) public AssertMatches( String patternString, boolean negateMatch ) { super( patternString, negateMatch ); } @ConstructorProperties({"patternString", "negateMatch", "delimiter"}) public AssertMatches( String patternString, boolean negateMatch, String delimiter ) { super( patternString, negateMatch, delimiter ); } @Override public boolean supportsPlannerLevel( PlannerLevel plannerLevel ) { return plannerLevel instanceof AssertionLevel; } @Override public void doAssert( FlowProcess flowProcess, ValueAssertionCall<Matcher> assertionCall ) { TupleEntry tupleEntry = assertionCall.getArguments(); if( matchWholeTuple( assertionCall.getContext(), tupleEntry ) ) BaseAssertion.throwFail( message, tupleEntry.getTuple().print(), patternString ); } }