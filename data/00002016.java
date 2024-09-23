public class JSONGetAllAggregateFunction extends NestedGetAllAggregateFunction<JsonNode, ArrayNode> { @ConstructorProperties({"stringRootPointer", "pointerMap"}) public JSONGetAllAggregateFunction( String stringRootPointer, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { this( stringRootPointer, false, pointerMap ); } @ConstructorProperties({"stringRootPointer", "failOnMissingNode", "pointerMap"}) public JSONGetAllAggregateFunction( String stringRootPointer, boolean failOnMissingNode, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { super( JSONCoercibleType.TYPE, stringRootPointer, failOnMissingNode, pointerMap ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "pointerMap"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { this( coercibleType, stringRootPointer, false, pointerMap ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "failOnMissingNode", "pointerMap"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, boolean failOnMissingNode, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { super( coercibleType, stringRootPointer, failOnMissingNode, pointerMap ); } @ConstructorProperties({"stringRootPointer", "streamWrapper", "pointerMap"}) public JSONGetAllAggregateFunction( String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { this( stringRootPointer, streamWrapper, false, pointerMap ); } @ConstructorProperties({"stringRootPointer", "streamWrapper", "failOnMissingNode", "pointerMap"}) public JSONGetAllAggregateFunction( String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, boolean failOnMissingNode, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { super( JSONCoercibleType.TYPE, stringRootPointer, streamWrapper, failOnMissingNode, pointerMap ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "streamWrapper", "pointerMap"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { this( coercibleType, stringRootPointer, streamWrapper, false, pointerMap ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "streamWrapper", "failOnMissingNode", "pointerMap"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, boolean failOnMissingNode, Map<String, NestedAggregate<JsonNode, ?>> pointerMap ) { super( coercibleType, stringRootPointer, streamWrapper, failOnMissingNode, pointerMap ); } @ConstructorProperties({"stringRootPointer", "streamWrapper", "fieldDeclaration", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( String stringRootPointer, Fields fieldDeclaration, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { this( stringRootPointer, fieldDeclaration, false, stringPointers, nestedAggregates ); } @ConstructorProperties({"stringRootPointer", "fieldDeclaration", "failOnMissingNode", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( String stringRootPointer, Fields fieldDeclaration, boolean failOnMissingNode, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { super( JSONCoercibleType.TYPE, stringRootPointer, fieldDeclaration, failOnMissingNode, stringPointers, nestedAggregates ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "fieldDeclaration", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, Fields fieldDeclaration, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { this( coercibleType, stringRootPointer, fieldDeclaration, false, stringPointers, nestedAggregates ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "fieldDeclaration", "failOnMissingNode", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, Fields fieldDeclaration, boolean failOnMissingNode, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { super( coercibleType, stringRootPointer, fieldDeclaration, failOnMissingNode, stringPointers, nestedAggregates ); } @ConstructorProperties({"stringRootPointer", "streamWrapper", "fieldDeclaration", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Fields fieldDeclaration, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { this( stringRootPointer, streamWrapper, fieldDeclaration, false, stringPointers, nestedAggregates ); } @ConstructorProperties({"stringRootPointer", "streamWrapper", "fieldDeclaration", "failOnMissingNode", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Fields fieldDeclaration, boolean failOnMissingNode, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { this( JSONCoercibleType.TYPE, stringRootPointer, streamWrapper, fieldDeclaration, failOnMissingNode, stringPointers, nestedAggregates ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "streamWrapper", "fieldDeclaration", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Fields fieldDeclaration, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { this( coercibleType, stringRootPointer, streamWrapper, fieldDeclaration, false, stringPointers, nestedAggregates ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "streamWrapper", "failOnMissingNode", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, boolean failOnMissingNode, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { super( coercibleType, stringRootPointer, streamWrapper, failOnMissingNode, stringPointers, nestedAggregates ); } @ConstructorProperties({"coercibleType", "stringRootPointer", "streamWrapper", "fieldDeclaration", "failOnMissingNode", "stringPointers", "nestedAggregates"}) public JSONGetAllAggregateFunction( JSONCoercibleType coercibleType, String stringRootPointer, SerFunction<Stream<JsonNode>, Stream<JsonNode>> streamWrapper, Fields fieldDeclaration, boolean failOnMissingNode, String[] stringPointers, NestedAggregate<JsonNode, ?>[] nestedAggregates ) { super( coercibleType, stringRootPointer, streamWrapper, fieldDeclaration, failOnMissingNode, stringPointers, nestedAggregates ); } }