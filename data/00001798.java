public class BalanceGroupGroupExpression extends RuleExpression { public BalanceGroupGroupExpression() { super( new SyncPipeExpressionGraph(), new ExpressionGraph() .arcs( new FlowElementExpression( Group.class ), new FlowElementExpression( Group.class ) ), new ExpressionGraph() .arc( new FlowElementExpression( ElementCapture.Primary, Pipe.class ), ScopeExpression.ANY, new FlowElementExpression( Group.class ) ) ); } }