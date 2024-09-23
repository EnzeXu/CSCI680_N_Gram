public class AndElementExpression extends ElementExpression { public static ElementExpression and ( String name , ElementExpression . . . elementMatchers ) { return new AndElementExpression ( name , elementMatchers ) ; } public static ElementExpression and ( String name , ElementCapture capture , ElementExpression . . . elementMatchers ) { return new AndElementExpression ( name , capture , elementMatchers ) ; } public static ElementExpression and ( ElementExpression . . . elementMatchers ) { return new AndElementExpression ( elementMatchers ) ; } public static ElementExpression and ( ElementCapture capture , ElementExpression . . . elementMatchers ) { return new AndElementExpression ( capture , elementMatchers ) ; } String name ; ElementExpression [ ] matchers ; public AndElementExpression ( String name , ElementExpression . . . matchers ) { this . matchers = matchers ; } public AndElementExpression ( String name , ElementCapture capture , ElementExpression . . . matchers ) { super ( capture ) ; this . matchers = matchers ; } public AndElementExpression ( ElementExpression . . . matchers ) { this . matchers = matchers ; } public AndElementExpression ( ElementCapture capture , ElementExpression . . . matchers ) { super ( capture ) ; this . matchers = matchers ; } @ Override public boolean applies ( PlannerContext plannerContext , ElementGraph elementGraph , FlowElement flowElement ) { for ( ElementExpression matcher : matchers ) { if ( !matcher . applies ( plannerContext , elementGraph , flowElement ) ) return false ; } return true ; } @ Override public String toString ( ) { if ( name != null ) return name ; final StringBuilder sb = new StringBuilder ( "And { " ) ; sb . append ( Arrays . toString ( matchers ) ) ; sb . append ( ' } ' ) ; return sb . toString ( ) ; } }