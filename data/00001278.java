public class TestHashJoinBlockingHashJoinExpression extends RuleExpression { public TestHashJoinBlockingHashJoinExpression() { super( new SyncPipeExpressionGraph(), new ExpressionGraph() .arc( new FlowElementExpression( HashJoin.class ), PathScopeExpression.BLOCKING, new FlowElementExpression( HashJoin.class ) ), new ExpressionGraph() .arc( new FlowElementExpression( ElementCapture.Primary, Pipe.class ), PathScopeExpression.BLOCKING, new FlowElementExpression( HashJoin.class ) ) ); } }