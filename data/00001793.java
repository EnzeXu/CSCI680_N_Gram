public class BalanceCheckpointWithTapExpression extends RuleExpression { public BalanceCheckpointWithTapExpression() { super( new ExpressionGraph() .arcs( new FlowElementExpression( ElementCapture.Primary, true, Checkpoint.class ), not( new FlowElementExpression( Tap.class ) ) ) ); } }