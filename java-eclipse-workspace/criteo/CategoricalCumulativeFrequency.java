import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Scanner;

public class CategoricalCumulativeFrequency {
	
	static String dataFile = "../../train_half.txt";

	public static void main(String[] args) throws FileNotFoundException {
		try (Scanner fieldsIn = new Scanner(new File("../../fields.txt"))) {
			String[] fieldNames = fieldsIn.nextLine().split("\t");
			System.out.println(Arrays.toString(fieldNames));
			for (int col = 0; col < fieldNames.length; col++) {
				String fieldName = fieldNames[col];
				if (fieldName.startsWith("C")) {
					System.out.println("Calculating frequency statistics for field " + fieldName);
					try (PrintWriter out = new PrintWriter(new FileWriter(fieldName + "-cum-freq.txt"))) {
						try (Scanner in = new Scanner(new File(dataFile))) {
							HashMap<String, Integer> countMap = new HashMap<>();
							int numRows = 0;
							while (in.hasNextLine()) {
								String[] row = in.nextLine().split("\t");
								String key = fieldName + ":" + (col < row.length ? row[col] : ""); // Note: Some rows come back with fewer than 40 columns.
								countMap.put(key, countMap.containsKey(key) ? countMap.get(key) + 1 : 1);
								numRows++;
							}
							List<Entry<String, Integer>> list = new ArrayList<>(countMap.entrySet());
					        list.sort(Entry.comparingByValue());
					        int totalCount = 0;
					        for (int i = list.size() - 1; i >= 0; i--) {
					        	Entry<String, Integer> entry = list.get(i);
					        	String key = entry.getKey();
					        	int value = entry.getValue();
					        	totalCount += value;
					        	out.printf("%s\t%d\t%d\t%f\n", key, value, totalCount, (double) totalCount / numRows);
					        }
					        out.close();
						}
					}
				}	
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}

}
