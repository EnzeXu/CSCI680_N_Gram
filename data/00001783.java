public class BalanceGroupMergeGroupExpression extends RuleExpression { public BalanceGroupMergeGroupExpression() { super( new SplicePipeExpressionGraph(), new ExpressionGraph() .arcs( new FlowElementExpression( Group.class ), new FlowElementExpression( Merge.class ), new FlowElementExpression( Group.class ) ), new ExpressionGraph() .arc( new FlowElementExpression( ElementCapture.Primary, Pipe.class ), ScopeExpression.ANY, new FlowElementExpression( Merge.class ) ) ); } }