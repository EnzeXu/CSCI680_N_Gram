public class StreamedSelfJoinSourcesExpressionGraph extends ExpressionGraph { public StreamedSelfJoinSourcesExpressionGraph ( ) { super ( SearchOrder . Depth , true ) ; FlowElementExpression intermediate = new FlowElementExpression ( HashJoin . class , TypeExpression . Topo . Linear ) ; this . arc ( or ( ElementCapture . Primary , new FlowElementExpression ( Tap . class , TypeExpression . Topo . LinearOut ) , new FlowElementExpression ( Boundary . class , TypeExpression . Topo . LinearOut ) , new FlowElementExpression ( Group . class , TypeExpression . Topo . LinearOut ) ) , PathScopeExpression . ALL , intermediate ) ; this . arc ( intermediate , PathScopeExpression . ALL , or ( new FlowElementExpression ( Tap . class ) , new FlowElementExpression ( Boundary . class ) , new FlowElementExpression ( Merge . class ) , new FlowElementExpression ( Group . class ) ) ) ; } }