import time
import sys
import collections
import traceback

TransformerStep = collections.namedtuple("TransformerStep", ["name", "transformer", "keys"])


class TransformerError(Exception):
    def __init__(self, exception, part):
        self.exception = exception
        self.part = part
    pass


class Transformer:
    @staticmethod
    def transform(data, steps: TransformerStep, verbose=True):
        updating_data = data

        step_cnt = 1
        for step in steps:
            name = step.name
            transformer = step.transformer

            if verbose:
                print("-> {} (step {}/{})".format(name, step_cnt, len(steps)))

            step_cnt += 1

            start = time.time()

            new_data = []
            doc_cnt = 0

            for doc in updating_data:
                new_doc = doc
                if verbose:
                    print('Document {}/{}'
                          .format(doc_cnt, len(updating_data), str(round(time.time() - start))),
                          end='\r')

                try:
                    for part in step.keys:
                        try:
                            transformation_result = transformer(doc[part])
                            new_doc[part] = transformation_result
                        except Exception as e:
                            raise TransformerError(exception=e,part=part)
                except TransformerError as e:
                    doc_cnt += 1
                    print("Cannot perform '{}' on document #{} ({}) because of exception:"
                          .format(name, doc_cnt, part))
                    traceback.print_tb(e.exception.__traceback__)
                    continue

                doc_cnt += 1
                new_data.append(new_doc)

            end = time.time()
            sys.stdout.flush()
            updating_data = new_data

            if verbose:
                print("Finished in " + str(round(end - start, 2)) + " sec\r")
                print("")

        return updating_data
