public class RemoveMalformedHashJoinPipelineTransformer extends RuleRemoveBranchTransformer { public RemoveMalformedHashJoinPipelineTransformer ( ) { super ( PostPipelines , new RuleExpression ( new MalformedJoinExpressionGraph ( ) ) ) ; } }