public class SyncPipeExpressionGraph extends ExpressionGraph { public SyncPipeExpressionGraph() { super( new AndElementExpression( ElementCapture.Primary, not( new FlowElementExpression( Extent.class ) ), not( new FlowElementExpression( Tap.class ) ), not( new FlowElementExpression( Boundary.class ) ), not( new FlowElementExpression( Checkpoint.class ) ), not( new FlowElementExpression( Group.class ) ), not( new FlowElementExpression( HashJoin.class ) ) ) ); } }