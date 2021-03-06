import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class MakeOneHotFreqCat {
	
	static int size = MostFrequentCategories.size;

	public static void main(String[] args) {
		// Get field names.
		try (Scanner fieldsIn = new Scanner(new File("../../fields.txt"))) {
			String[] fieldNames = fieldsIn.nextLine().split("\t");
//			System.out.println(Arrays.toString(fieldNames));

			// Get most frequent categories
			HashMap<String, Integer> catMap = new HashMap<>();
			try (Scanner catsIn = new Scanner(new File("../../1000_most_frequent_categories.txt"))) {
				int i = 0;
				while (catsIn.hasNextLine())
					catMap.put(catsIn.nextLine(), i++);
			}
			
			// Transform lines from stdin (Criteo format) to sparse one-hot-encoded lines on stdout
			int firstCatIdx = 0;
			while (!fieldNames[firstCatIdx].startsWith("C"))
				firstCatIdx++;
			try (Scanner in = new Scanner(System.in)) {
				while (in.hasNextLine()) {
					String[] data = in.nextLine().split("\t");
					for (int i = 0; i < firstCatIdx; i++)
						System.out.print(data[i] + ",");
					ArrayList<String> oneHots = new ArrayList<String>();
					for (int i = firstCatIdx; i < data.length; i++) {
						String cat = fieldNames[i] + ":" + data[i];
						if (catMap.containsKey(cat))
							oneHots.add(cat);
					}
					System.out.println(String.join(",", oneHots.toArray(new String[oneHots.size()])));
//					// Dense binary:
//					int[] oneHots = new int[catMap.size()];
//					for (int i = firstCatIdx; i < data.length; i++) {
//						String cat = fieldNames[i] + ":" + data[i];
//						if (catMap.containsKey(cat))
//							oneHots[catMap.get(cat)] = 1;
//					}
//					for (int i = 0; i < oneHots.length; i++) 
//						System.out.print(oneHots[i] + (i < oneHots.length - 1 ? "," : "\n"));
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}

}
