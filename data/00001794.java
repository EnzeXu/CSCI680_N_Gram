public class BalanceHashJoinToHashJoinExpression extends RuleExpression { public BalanceHashJoinToHashJoinExpression() { super( new SyncPipeExpressionGraph(), new ExpressionGraph() .arc( new FlowElementExpression( HashJoin.class ), PathScopeExpression.ANY, new FlowElementExpression( HashJoin.class ) ), new ExpressionGraph() .arc( new FlowElementExpression( ElementCapture.Primary, Pipe.class ), PathScopeExpression.ANY, new FlowElementExpression( HashJoin.class ) ) ); } }