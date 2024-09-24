public class JSONGetFunction extends NestedGetFunction<JsonNode, ArrayNode> { @ConstructorProperties("pointerMap") public JSONGetFunction( Map<Fields, String> pointerMap ) { this( asFields( pointerMap.keySet() ), asArray( pointerMap.values() ) ); } @ConstructorProperties({"pointerMap", "failOnMissingNode"}) public JSONGetFunction( Map<Fields, String> pointerMap, boolean failOnMissingNode ) { this( asFields( pointerMap.keySet() ), failOnMissingNode, asArray( pointerMap.values() ) ); } @ConstructorProperties({"fieldDeclaration", "stringPointers"}) public JSONGetFunction( Fields fieldDeclaration, String... stringPointers ) { this( fieldDeclaration, false, stringPointers ); } @ConstructorProperties({"fieldDeclaration", "failOnMissingNode", "stringPointers"}) public JSONGetFunction( Fields fieldDeclaration, boolean failOnMissingNode, String... stringPointers ) { super( JSONCoercibleType.TYPE, fieldDeclaration, failOnMissingNode, stringPointers ); } @ConstructorProperties({"coercibleType", "pointerMap"}) public JSONGetFunction( JSONCoercibleType coercibleType, Map<Fields, String> pointerMap ) { this( coercibleType, asFields( pointerMap.keySet() ), asArray( pointerMap.values() ) ); } @ConstructorProperties({"coercibleType", "pointerMap", "failOnMissingNode"}) public JSONGetFunction( JSONCoercibleType coercibleType, Map<Fields, String> pointerMap, boolean failOnMissingNode ) { this( coercibleType, asFields( pointerMap.keySet() ), failOnMissingNode, asArray( pointerMap.values() ) ); } @ConstructorProperties({"coercibleType", "fieldDeclaration", "stringPointers"}) public JSONGetFunction( JSONCoercibleType coercibleType, Fields fieldDeclaration, String... stringPointers ) { this( coercibleType, fieldDeclaration, false, stringPointers ); } @ConstructorProperties({"coercibleType", "fieldDeclaration", "failOnMissingNode", "stringPointers"}) public JSONGetFunction( JSONCoercibleType coercibleType, Fields fieldDeclaration, boolean failOnMissingNode, String... stringPointers ) { super( coercibleType, fieldDeclaration, failOnMissingNode, stringPointers ); } }