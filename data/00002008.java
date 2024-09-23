public class JSONTypeTest { @Test public void stringLiteralCoercions() { testCoercion( "\"Foo\"", JsonNodeType.STRING, "Foo", String.class ); testCoercion( "Foo", JsonNodeType.STRING, "Foo", String.class ); testCoercion( "100", JsonNodeType.NUMBER, 100, Integer.class ); testCoercion( 100, JsonNodeType.NUMBER, 100, Integer.class ); testCoercion( "Foo", JsonNodeType.STRING, JSONCoercibleType.TYPE.canonical( "Foo" ), JsonNode.class ); } private void testCoercion( Object value, JsonNodeType nodeType, Object resultValue, Class resultType ) { JsonNode canonical = JSONCoercibleType.TYPE.canonical( value ); assertEquals( nodeType, canonical.getNodeType() ); assertEquals( resultValue, JSONCoercibleType.TYPE.coerce( canonical, resultType ) ); } @Test public void objectCoercions() { for( String value : JSONData.objects ) testContainerCoercion( value, JsonNodeType.OBJECT, String.class ); } @Test public void arrayCoercions() { for( String value : JSONData.arrays ) testContainerCoercion( value, JsonNodeType.ARRAY, String.class ); } @Test public void mapCoercions() { Map<String, Object> map = new LinkedHashMap<>(); map.put( "name", "John Doe" ); map.put( "list", Arrays.asList( "John", "Jane" ) ); JsonNode canonical = JSONCoercibleType.TYPE.canonical( map ); assertEquals( JsonNodeType.OBJECT, canonical.getNodeType() ); assertEquals( map, JSONCoercibleType.TYPE.coerce( canonical, Map.class ) ); } @Test public void listCoercions() { List<Object> list = new LinkedList<>(); list.add( "John Doe" ); list.add( Arrays.asList( "John", "Jane" ) ); JsonNode canonical = JSONCoercibleType.TYPE.canonical( list ); assertEquals( JsonNodeType.ARRAY, canonical.getNodeType() ); assertEquals( list, JSONCoercibleType.TYPE.coerce( canonical, List.class ) ); } @Test public void pojoCoercions() { ObjectMapper mapper = new ObjectMapper(); mapper.registerModule( new JavaTimeModule() ); JSONCoercibleType type = new JSONCoercibleType( mapper ); Instant instant = Instant.ofEpochSecond( 1525456424, 337000000 ); JsonNode canonical = type.canonical( instant ); String coerce = type.coerce( canonical, String.class ); assertEquals( "1525456424.337000000", coerce ); } @Test public void pojoCoercionsReversed() { ObjectMapper mapper = new ObjectMapper(); mapper.registerModule( new JavaTimeModule() ); JSONCoercibleType type = new JSONCoercibleType( mapper ); Instant instant = Instant.ofEpochSecond( 1525456424, 337000000 ); JsonNode canonical = type.canonical( instant ); Instant coerce = type.coerce( canonical, Instant.class ); assertEquals( instant, coerce ); } private void testContainerCoercion( String value, JsonNodeType nodeType, Class resultType ) { JsonNode canonical = JSONCoercibleType.TYPE.canonical( value ); assertEquals( nodeType, canonical.getNodeType() ); assertEquals( value.replaceAll( "\\s", "" ), JSONCoercibleType.TYPE.coerce( canonical, resultType ) ); } @Test public void stringCoercions() { for( String value : JSONData.objects ) testContainerStringCoercion( value, JsonNodeType.OBJECT, String.class ); } private void testContainerStringCoercion( String value, JsonNodeType nodeType, Class resultType ) { JsonNode canonical = (JsonNode) Coercions.coerce( Coercions.STRING, value, JSONCoercibleType.TYPE ); assertEquals( nodeType, canonical.getNodeType() ); assertEquals( value.replaceAll( "\\s", "" ), JSONCoercibleType.TYPE.coerce( canonical, resultType ) ); } }