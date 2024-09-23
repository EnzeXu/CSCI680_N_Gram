public class ExpressionFilter extends ExpressionOperation implements Filter < ScriptOperation . Context > { @ ConstructorProperties ( { "expression" } ) public ExpressionFilter ( String expression ) { super ( expression ) ; } @ ConstructorProperties ( { "expression" , "parameterType" } ) public ExpressionFilter ( String expression , Class parameterType ) { super ( expression , parameterType ) ; } @ ConstructorProperties ( { "expression" , "parameterNames" , "parameterTypes" } ) public ExpressionFilter ( String expression , String [ ] parameterNames , Class [ ] parameterTypes ) { super ( expression , parameterNames , parameterTypes ) ; } @ Override public boolean isRemove ( FlowProcess flowProcess , FilterCall < Context > filterCall ) { return ( Boolean ) evaluate ( filterCall . getContext ( ) , filterCall . getArguments ( ) ) ; } }