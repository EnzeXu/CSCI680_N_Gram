public class NoOpPipeExpressionGraph extends ExpressionGraph { public NoOpPipeExpressionGraph() { super( OrElementExpression.or( ElementCapture.Primary, new FlowElementExpression( true, Pipe.class ), new FlowElementExpression( true, Checkpoint.class ), new FlowElementExpression( false, SubAssembly.class ) ) ); } }