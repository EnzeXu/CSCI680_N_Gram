public class DirectMappedCacheTest { class TestCacheEvictionCallBack implements CacheEvictionCallback < Object , Object > { int evictionCounter = 0 ; List < Map . Entry < Object , Object > > evictedEntries = new ArrayList < Map . Entry < Object , Object > > ( ) ; @ Override public void evict ( Map . Entry < Object , Object > entry ) { assertNotNull ( entry ) ; evictionCounter++ ; evictedEntries . add ( entry ) ; } } @ Test public void testDirectMappedCache ( ) { CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , CacheEvictionCallback . NULL ) ; assertNotNull ( cache ) ; assertEquals ( 0 , cache . size ( ) ) ; String key = "abc" ; String value = "def" ; cache . put ( key , value ) ; assertEquals ( 1 , cache . size ( ) ) ; assertTrue ( cache . containsKey ( key ) ) ; assertTrue ( cache . containsValue ( value ) ) ; assertSame ( value , cache . get ( key ) ) ; Set < Object > keys = cache . keySet ( ) ; assertEquals ( 1 , keys . size ( ) ) ; assertTrue ( keys . contains ( key ) ) ; Collection < Object > values = cache . values ( ) ; assertTrue ( values . contains ( value ) ) ; cache . clear ( ) ; assertEquals ( 0 , cache . size ( ) ) ; assertFalse ( cache . containsKey ( key ) ) ; assertFalse ( cache . containsValue ( value ) ) ; } class Collider { private final String value ; Collider ( String value ) { this . value = value ; } @ Override public int hashCode ( ) { return 42 ; } @ Override public boolean equals ( Object object ) { if ( this == object ) return true ; Collider collider = ( Collider ) object ; if ( value != null ? !value . equals ( collider . value ) : collider . value != null ) { return false ; } return true ; } } @ Test public void testDirectMappedCacheEviction ( ) { TestCacheEvictionCallBack callBack = new TestCacheEvictionCallBack ( ) ; CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , callBack ) ; Collider key = new Collider ( "key" ) ; String value = "value" ; cache . put ( key , value ) ; assertEquals ( 1 , cache . size ( ) ) ; assertEquals ( 0 , callBack . evictionCounter ) ; Collider secondKey = new Collider ( "anotherKey" ) ; cache . put ( secondKey , value ) ; assertEquals ( 1 , cache . size ( ) ) ; assertEquals ( 1 , callBack . evictionCounter ) ; assertTrue ( cache . containsKey ( secondKey ) ) ; assertTrue ( cache . containsValue ( value ) ) ; assertSame ( value , cache . get ( secondKey ) ) ; assertTrue ( callBack . evictedEntries . get ( 0 ) . getKey ( ) . equals ( key ) ) ; assertTrue ( callBack . evictedEntries . get ( 0 ) . getValue ( ) . equals ( value ) ) ; } @ Test public void testMaxCapacity ( ) { TestCacheEvictionCallBack callBack = new TestCacheEvictionCallBack ( ) ; CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , callBack ) ; for ( int i = 0 ; i < cache . getCapacity ( ) + 10 ; i++ ) cache . put ( i , i ) ; assertEquals ( cache . getCapacity ( ) , cache . size ( ) ) ; assertEquals ( 10 , callBack . evictionCounter ) ; } @ Test ( expected = IllegalArgumentException . class ) public void testNullKey ( ) { CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , CacheEvictionCallback . NULL ) ; cache . put ( null , "a" ) ; } @ Test ( expected = IllegalArgumentException . class ) public void testNullValue ( ) { CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , CacheEvictionCallback . NULL ) ; cache . put ( "a" , null ) ; } @ Test ( expected = IllegalArgumentException . class ) public void testContainsKeyNull ( ) { CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , CacheEvictionCallback . NULL ) ; cache . containsKey ( null ) ; } @ Test ( expected = IllegalArgumentException . class ) public void testContainsValueNull ( ) { CascadingCache < Object , Object > cache = getDirectMappedCache ( 10 , CacheEvictionCallback . NULL ) ; cache . containsValue ( null ) ; } @ Test ( expected = IllegalStateException . class ) public void testCreateCacheNegative ( ) { getDirectMappedCache ( -1 , CacheEvictionCallback . NULL ) ; } private CascadingCache < Object , Object > getDirectMappedCache ( int capacity , CacheEvictionCallback cacheEvictionCallback ) { CascadingCache < Object , Object > map = new DirectMappedCache < Object , Object > ( ) ; map . setCacheEvictionCallback ( cacheEvictionCallback ) ; map . setCapacity ( capacity ) ; map . initialize ( ) ; return map ; } }