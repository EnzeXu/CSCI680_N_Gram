public class NoOpPipeMultiGraphExpression extends RuleExpression { public static final ElementExpression SPLIT = or ( new FlowElementExpression ( Pipe . class ) , new FlowElementExpression ( Tap . class ) ) ; public static final ElementExpression JOIN = or ( new FlowElementExpression ( Pipe . class ) , new FlowElementExpression ( Tap . class ) ) ; public NoOpPipeMultiGraphExpression ( ) { super ( new ConsecutiveNoOpPipesExpressionGraph ( ) , new ExpressionGraph ( ) . arcs ( SPLIT , new FlowElementExpression ( ElementCapture . Primary , true , Pipe . class ) , JOIN ) . arcs ( SPLIT , new FlowElementExpression ( ElementCapture . Secondary , true , Pipe . class ) , JOIN ) ) ; } }