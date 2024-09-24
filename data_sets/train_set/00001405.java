public class SetMultiMap<K, V> extends MultiMap<Set<V>, K, V> { @Override protected Map<K, Set<V>> createMap() { return new HashMap<>(); } protected Set<V> createCollection() { return new LinkedHashSet<>(); } protected Set<V> emptyCollection() { return Collections.emptySet(); } }