public class ConsecutiveNoOpPipesExpressionGraph extends ExpressionGraph { public ConsecutiveNoOpPipesExpressionGraph ( ) { super ( SearchOrder . ReverseTopological ) ; this . arc ( new FlowElementExpression ( true , Pipe . class ) , ScopeExpression . ALL , new FlowElementExpression ( ElementCapture . Primary , true , Pipe . class ) ) ; } }