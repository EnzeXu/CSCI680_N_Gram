public class JSONBuildAsAggregator extends NestedBaseBuildAggregator<JsonNode, ArrayNode> { @ConstructorProperties({"fieldDeclaration", "buildSpecs"}) public JSONBuildAsAggregator( Fields fieldDeclaration, BuildSpec... buildSpecs ) { super( JSONCoercibleType.TYPE, fieldDeclaration, buildSpecs ); } @ConstructorProperties({"coercibleType", "fieldDeclaration", "buildSpecs"}) public JSONBuildAsAggregator( JSONCoercibleType coercibleType, Fields fieldDeclaration, BuildSpec... buildSpecs ) { super( coercibleType, fieldDeclaration, buildSpecs ); } @Override protected boolean isInto() { return false; } }