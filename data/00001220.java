public class LRUHashMapCacheFactory extends BaseCacheFactory { @ Override public CascadingCache create ( FlowProcess flowProcess ) { return new LRUHashMapCache ( ) ; } }