public class TestCheckpointExpression extends RuleExpression { public TestCheckpointExpression() { super( new ExpressionGraph() .arcs( new FlowElementExpression( ElementCapture.Primary, true, Checkpoint.class ), not( new FlowElementExpression( Tap.class ) ) ) ); } }